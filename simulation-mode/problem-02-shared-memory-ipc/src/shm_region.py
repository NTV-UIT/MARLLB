"""
Shared Memory Region - Python Implementation (Cross-platform)

Author: MARLLB Implementation Team
Date: December 2025
"""

import mmap
import os
import sys
import time
import warnings
import tempfile
from typing import Optional, Dict, List

from shm_layout import (
    MessageOutLayout, MessageInLayout, RingBufferLayout,
    MAX_AS, RING_BUFFER_SIZE, TOTAL_SHM_SIZE
)

# Platform detection
IS_LINUX = sys.platform.startswith('linux')
IS_MACOS = sys.platform == 'darwin'


def get_shm_path(name: str) -> str:
    """Get platform-specific shared memory path."""
    if IS_LINUX:
        return f"/dev/shm/{name}"
    elif IS_MACOS:
        return f"/tmp/marllb_shm_{name}"
    else:
        return os.path.join(tempfile.gettempdir(), f"marllb_shm_{name}")


class SharedMemoryRegion:
    """High-level interface for MARLLB shared memory communication."""
    
    def __init__(self, name: str, mm: mmap.mmap, fd: int, owner: bool = False):
        self.name = name
        self.mm = mm
        self.fd = fd
        self.owner = owner
        
        self.last_read_seq = 0
        self.last_write_seq = 0
        
        self.msg_out_ring_offset = 0
        self.msg_in_offset = RingBufferLayout.TOTAL_SIZE
    
    @classmethod
    def create(cls, name: str, size: Optional[int] = None):
        """Create new shared memory region."""
        if size is None:
            size = TOTAL_SHM_SIZE
        
        shm_path = get_shm_path(name)
        
        if os.path.exists(shm_path):
            warnings.warn(f"Shared memory {name} already exists, removing...")
            os.unlink(shm_path)
        
        fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        mm.seek(0)
        mm.write(b'\x00' * size)
        
        print(f"✓ Created shared memory: {shm_path} ({size} bytes)")
        return cls(name, mm, fd, owner=True)
    
    @classmethod
    def attach(cls, name: str):
        """Attach to existing shared memory region."""
        shm_path = get_shm_path(name)
        
        if not os.path.exists(shm_path):
            raise FileNotFoundError(f"Shared memory {name} not found at {shm_path}")
        
        fd = os.open(shm_path, os.O_RDWR)
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        
        print(f"✓ Attached to shared memory: {shm_path} ({size} bytes)")
        return cls(name, mm, fd, owner=False)
    
    def write_observation(self, sequence_id: int, timestamp_us: Optional[int] = None,
                         active_servers: List[int] = None, server_stats: Dict = None) -> int:
        """Write observation to ring buffer (VPP → RL)."""
        if timestamp_us is None:
            timestamp_us = int(time.time() * 1e6)
        if active_servers is None:
            active_servers = []
        if server_stats is None:
            server_stats = {}
        
        active_bitmap = sum(1 << sid for sid in active_servers)
        
        as_stats = [server_stats.get(i, {'n_flow_on': 0, 'reservoir_features': [0.0] * 10})
                   for i in range(MAX_AS)]
        
        packed = MessageOutLayout.pack(sequence_id, timestamp_us, active_bitmap,
                                      len(active_servers), as_stats)
        
        write_idx = RingBufferLayout.get_write_index(self.mm)
        slot = write_idx % RING_BUFFER_SIZE
        
        offset = RingBufferLayout.get_message_offset(slot)
        self.mm.seek(offset)
        self.mm.write(packed)
        
        RingBufferLayout.set_write_index(self.mm, write_idx + 1)
        self.last_write_seq = sequence_id
        
        return slot
    
    def read_observation(self, slot: Optional[int] = None) -> Optional[Dict]:
        """Read observation from ring buffer (RL agent side)."""
        write_idx = RingBufferLayout.get_write_index(self.mm)
        
        if write_idx == 0:
            return None
        
        if slot is None:
            slot = (write_idx - 1) % RING_BUFFER_SIZE
        
        offset = RingBufferLayout.get_message_offset(slot)
        self.mm.seek(offset)
        data = self.mm.read(MessageOutLayout.MESSAGE_SIZE)
        
        obs = MessageOutLayout.unpack(data)
        
        if obs['sequence_id'] <= self.last_read_seq:
            return None
        
        if obs['sequence_id'] > self.last_read_seq + 1:
            missed = obs['sequence_id'] - self.last_read_seq - 1
            warnings.warn(f"Missed {missed} observations")
        
        self.last_read_seq = obs['sequence_id']
        return obs
    
    def write_action(self, sequence_id: int, weights: List[float], alias_table=None):
        """Write action message (RL → VPP)."""
        packed = MessageInLayout.pack(sequence_id, weights, alias_table)
        self.mm.seek(self.msg_in_offset)
        self.mm.write(packed)
        self.last_write_seq = sequence_id
    
    def read_action(self) -> Optional[Dict]:
        """Read action message (VPP side)."""
        self.mm.seek(self.msg_in_offset)
        data = self.mm.read(MessageInLayout.MESSAGE_SIZE)
        action = MessageInLayout.unpack(data)
        
        if action['sequence_id'] <= self.last_read_seq:
            return None
        
        self.last_read_seq = action['sequence_id']
        return action
    
    def close(self):
        """Close shared memory."""
        if self.mm:
            self.mm.close()
            self.mm = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
    
    def unlink(self):
        """Unlink shared memory."""
        shm_path = get_shm_path(self.name)
        if os.path.exists(shm_path):
            os.unlink(shm_path)
            print(f"✓ Unlinked shared memory: {shm_path}")
    
    def __del__(self):
        self.close()
        if self.owner:
            self.unlink()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if self.owner:
            self.unlink()
    
    def __repr__(self):
        return f"SharedMemoryRegion(name='{self.name}')"


if __name__ == "__main__":
    print("=== Shared Memory Region Test ===\n")
    
    with SharedMemoryRegion.create("test_marllb") as shm:
        print(f"  {shm}\n")
        
        # Write observation
        server_stats = {
            0: {'n_flow_on': 10, 'reservoir_features': [0.1, 0.2, 0.05, 0.11, 0.21,
                                                        1.0, 1.5, 0.3, 1.1, 1.6]},
            1: {'n_flow_on': 15, 'reservoir_features': [0.15, 0.25, 0.06, 0.16, 0.26,
                                                        1.2, 1.7, 0.4, 1.3, 1.8]}
        }
        
        shm.write_observation(sequence_id=1, active_servers=[0, 1], server_stats=server_stats)
        print("✓ Written observation (seq=1)\n")
        
        # Read observation
        obs = shm.read_observation()
        if obs:
            print(f"Sequence ID: {obs['sequence_id']}")
            print(f"Active servers: {obs['active_servers']}")
            for sid in obs['active_servers']:
                stats = obs['server_stats'][sid]
                print(f"  Server {sid}: flows={stats['n_flow_on']}, "
                      f"FCT={stats['fct_mean']:.3f}s")
        
        # Write action
        shm.write_action(sequence_id=1, weights=[1.0, 1.5])
        print("\n✓ Written action (seq=1)")
        
        # Read action
        action = shm.read_action()
        if action:
            print(f"  Weights: {action['weights']}")
    
    print("\n✅ Test completed!")
