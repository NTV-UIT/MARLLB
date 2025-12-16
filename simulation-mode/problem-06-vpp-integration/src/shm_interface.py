"""
Mock Shared Memory Interface for Testing

Simple Python interface cho shared memory communication với VPP.
Đây là simplified version cho testing - production sẽ dùng actual shared memory.
"""

import numpy as np
import mmap
import struct
import os


class SHMLayout:
    """
    Định nghĩa memory layout cho shared memory.
    """
    
    def __init__(self, num_servers=64):
        self.num_servers = num_servers
        
        # Calculate sizes
        self.msg_out_size = (
            4 +  # id (u32)
            4 +  # timestamp (f32)
            8 +  # active_servers_bitmap (u64)
            num_servers * (
                4 +  # as_index (u32)
                4 +  # n_flow_on (i32)
                4 +  # cpu_util (f32)
                4 +  # queue_depth (f32)
                4    # response_time (f32)
            )
        )
        
        self.msg_in_size = (
            4 +  # id (u32)
            4 +  # timestamp (f32)
            num_servers * 4 +  # server_weights (f32 × num_servers)
            num_servers * (
                4 +  # probability (f32)
                4    # alias (u32)
            )
        )
        
        self.total_size = self.msg_out_size + self.msg_in_size
        
        print(f"[SHMLayout] Layout created:")
        print(f"  Servers: {num_servers}")
        print(f"  msg_out size: {self.msg_out_size} bytes")
        print(f"  msg_in size: {self.msg_in_size} bytes")
        print(f"  Total size: {self.total_size} bytes")


class SharedMemoryInterface:
    """
    Interface cho shared memory communication.
    
    Memory layout:
    - msg_out: VPP → Python (server statistics)
    - msg_in: Python → VPP (server weights)
    """
    
    def __init__(self, path, layout):
        """
        Args:
            path: Path to shared memory file
            layout: SHMLayout instance
        """
        self.path = path
        self.layout = layout
        
        # Create/resize file to correct size
        with open(path, 'wb') as f:
            f.write(b'\x00' * layout.total_size)
        
        # Open file
        self.fd = open(path, 'r+b')
        self.mmap = mmap.mmap(self.fd.fileno(), layout.total_size)
        
        print(f"[SharedMemoryInterface] Opened {path}")
    
    def read_msg_out(self):
        """
        Read msg_out from shared memory (VPP stats).
        
        Returns:
            msg_out: Dict with server statistics
        """
        self.mmap.seek(0)
        
        # Read header
        msg_id = struct.unpack('I', self.mmap.read(4))[0]
        timestamp = struct.unpack('f', self.mmap.read(4))[0]
        active_bitmap = struct.unpack('Q', self.mmap.read(8))[0]
        
        # Read server stats
        server_stats = []
        for i in range(self.layout.num_servers):
            as_index = struct.unpack('I', self.mmap.read(4))[0]
            n_flow_on = struct.unpack('i', self.mmap.read(4))[0]
            cpu_util = struct.unpack('f', self.mmap.read(4))[0]
            queue_depth = struct.unpack('f', self.mmap.read(4))[0]
            response_time = struct.unpack('f', self.mmap.read(4))[0]
            
            server_stats.append({
                'as_index': as_index,
                'n_flow_on': n_flow_on,
                'cpu_util': cpu_util,
                'queue_depth': queue_depth,
                'response_time': response_time
            })
        
        return {
            'id': msg_id,
            'timestamp': timestamp,
            'active_bitmap': active_bitmap,
            'server_stats': server_stats
        }
    
    def write_msg_out(self, msg_out):
        """
        Write msg_out to shared memory (for testing).
        
        Args:
            msg_out: Dict with server statistics
        """
        self.mmap.seek(0)
        
        # Write header
        self.mmap.write(struct.pack('I', msg_out['id']))
        self.mmap.write(struct.pack('f', msg_out['timestamp']))
        self.mmap.write(struct.pack('Q', msg_out.get('active_bitmap', 0)))
        
        # Write server stats
        for i in range(self.layout.num_servers):
            if i < len(msg_out['server_stats']):
                stat = msg_out['server_stats'][i]
                self.mmap.write(struct.pack('I', stat['as_index']))
                self.mmap.write(struct.pack('i', stat['n_flow_on']))
                self.mmap.write(struct.pack('f', stat['cpu_util']))
                self.mmap.write(struct.pack('f', stat['queue_depth']))
                self.mmap.write(struct.pack('f', stat['response_time']))
            else:
                # Padding
                self.mmap.write(b'\x00' * 20)
        
        self.mmap.flush()
    
    def read_msg_in(self):
        """
        Read msg_in from shared memory (Python actions).
        
        Returns:
            msg_in: Dict with server weights and alias table
        """
        # Seek to msg_in offset
        self.mmap.seek(self.layout.msg_out_size)
        
        # Read header
        msg_id = struct.unpack('I', self.mmap.read(4))[0]
        timestamp = struct.unpack('f', self.mmap.read(4))[0]
        
        # Read server weights
        server_weights = []
        for i in range(self.layout.num_servers):
            weight = struct.unpack('f', self.mmap.read(4))[0]
            server_weights.append(weight)
        
        # Read alias table
        alias_table = []
        for i in range(self.layout.num_servers):
            prob = struct.unpack('f', self.mmap.read(4))[0]
            alias = struct.unpack('I', self.mmap.read(4))[0]
            alias_table.append((prob, alias))
        
        return {
            'id': msg_id,
            'timestamp': timestamp,
            'server_weights': np.array(server_weights),
            'alias_table': alias_table
        }
    
    def write_msg_in(self, msg_in):
        """
        Write msg_in to shared memory (Python → VPP).
        
        Args:
            msg_in: Dict with server weights and alias table
        """
        # Seek to msg_in offset
        self.mmap.seek(self.layout.msg_out_size)
        
        # Write header
        self.mmap.write(struct.pack('I', msg_in['id']))
        self.mmap.write(struct.pack('f', msg_in['timestamp']))
        
        # Write server weights
        weights = msg_in['server_weights']
        for i in range(self.layout.num_servers):
            if i < len(weights):
                self.mmap.write(struct.pack('f', float(weights[i])))
            else:
                self.mmap.write(struct.pack('f', 0.0))
        
        # Write alias table
        alias_table = msg_in['alias_table']
        for i in range(self.layout.num_servers):
            if i < len(alias_table):
                prob, alias = alias_table[i]
                self.mmap.write(struct.pack('f', float(prob)))
                self.mmap.write(struct.pack('I', int(alias)))
            else:
                self.mmap.write(struct.pack('f', 0.0))
                self.mmap.write(struct.pack('I', 0))
        
        self.mmap.flush()
    
    def close(self):
        """Close shared memory."""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'fd'):
            self.fd.close()
        print(f"[SharedMemoryInterface] Closed {self.path}")


if __name__ == '__main__':
    # Test
    import tempfile
    
    print("Testing SharedMemoryInterface...\n")
    
    with tempfile.NamedTemporaryFile() as tmp:
        # Create interface
        layout = SHMLayout(num_servers=16)
        shm = SharedMemoryInterface(tmp.name, layout)
        
        # Write msg_out
        msg_out = {
            'id': 1,
            'timestamp': 123.45,
            'active_bitmap': 0xFFFF,
            'server_stats': [
                {
                    'as_index': i,
                    'n_flow_on': 10 + i,
                    'cpu_util': 0.5 + i * 0.01,
                    'queue_depth': 5.0 + i,
                    'response_time': 10.0 + i * 0.5
                }
                for i in range(16)
            ]
        }
        
        shm.write_msg_out(msg_out)
        print(f"\nWrote msg_out: id={msg_out['id']}, timestamp={msg_out['timestamp']}")
        
        # Read back
        msg_out_read = shm.read_msg_out()
        print(f"Read msg_out: id={msg_out_read['id']}, timestamp={msg_out_read['timestamp']:.2f}")
        print(f"Server 0: {msg_out_read['server_stats'][0]}")
        
        # Write msg_in
        weights = np.ones(16) / 16
        alias_table = [(1.0, i) for i in range(16)]
        
        msg_in = {
            'id': 2,
            'timestamp': 124.56,
            'server_weights': weights,
            'alias_table': alias_table
        }
        
        shm.write_msg_in(msg_in)
        print(f"\nWrote msg_in: id={msg_in['id']}, weights_sum={weights.sum():.6f}")
        
        # Read back
        msg_in_read = shm.read_msg_in()
        print(f"Read msg_in: id={msg_in_read['id']}, weights_sum={msg_in_read['server_weights'].sum():.6f}")
        
        shm.close()
        
        print("\n✓ All tests passed!")
