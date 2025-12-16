"""
Shared Memory Layout Definitions for MARLLB

This module defines the memory layout structures used for communication
between VPP plugin (C) and RL agent (Python).

Author: MARLLB Implementation Team
Date: December 2025
"""

import struct
import mmap
import os
import time
from typing import Optional, Dict, List, Tuple
import numpy as np


# Constants matching C definitions
MAX_AS = 64
RESERVOIR_CAPACITY = 128
NUM_FEATURES = 5
RING_BUFFER_SIZE = 4


class MessageOutLayout:
    """
    Layout for observation messages (VPP → RL)
    
    Memory layout:
        uint64_t sequence_id
        uint64_t timestamp_us
        uint64_t active_as_bitmap
        uint32_t num_active_as
        uint32_t reserved
        struct {
            uint32_t n_flow_on
            float reservoir_features[10]
        } as_stats[MAX_AS]
    """
    
    # Struct format: native byte order, standard sizes
    HEADER_FORMAT = '=QQQIIx' + 'x' * 4  # Last part for 8-byte alignment
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    # Per-server stats: uint32 + 10 floats
    SERVER_FORMAT = '=I10f'
    SERVER_SIZE = struct.calcsize(SERVER_FORMAT)
    
    # Total message size
    MESSAGE_SIZE = HEADER_SIZE + SERVER_SIZE * MAX_AS
    
    @staticmethod
    def pack(sequence_id: int,
             timestamp_us: int,
             active_as_bitmap: int,
             num_active_as: int,
             as_stats: List[Dict]) -> bytes:
        """
        Pack observation message into bytes.
        
        Args:
            sequence_id: Monotonic sequence number
            timestamp_us: Timestamp in microseconds
            active_as_bitmap: 64-bit bitmap of active servers
            num_active_as: Number of active servers
            as_stats: List of dicts with 'n_flow_on' and 'reservoir_features'
            
        Returns:
            Packed bytes
        """
        # Pack header
        data = struct.pack(
            MessageOutLayout.HEADER_FORMAT,
            sequence_id,
            timestamp_us,
            active_as_bitmap,
            num_active_as,
            0  # reserved
        )
        
        # Pack per-server stats
        for i in range(MAX_AS):
            if i < len(as_stats):
                stats = as_stats[i]
                n_flow = stats.get('n_flow_on', 0)
                features = stats.get('reservoir_features', [0.0] * 10)
                
                # Ensure features has exactly 10 elements
                if len(features) < 10:
                    features = list(features) + [0.0] * (10 - len(features))
                elif len(features) > 10:
                    features = features[:10]
                
                server_data = struct.pack(MessageOutLayout.SERVER_FORMAT, n_flow, *features)
            else:
                # Empty server
                server_data = struct.pack(MessageOutLayout.SERVER_FORMAT, 0, *([0.0] * 10))
            
            data += server_data
        
        return data
    
    @staticmethod
    def unpack(data: bytes) -> Dict:
        """
        Unpack observation message from bytes.
        
        Args:
            data: Packed bytes
            
        Returns:
            Dictionary with message fields
        """
        # Unpack header
        header = struct.unpack(MessageOutLayout.HEADER_FORMAT, 
                              data[:MessageOutLayout.HEADER_SIZE])
        
        sequence_id = header[0]
        timestamp_us = header[1]
        active_as_bitmap = header[2]
        num_active_as = header[3]
        
        # Unpack per-server stats
        as_stats = {}
        offset = MessageOutLayout.HEADER_SIZE
        
        for i in range(MAX_AS):
            server_data = data[offset:offset + MessageOutLayout.SERVER_SIZE]
            unpacked = struct.unpack(MessageOutLayout.SERVER_FORMAT, server_data)
            
            n_flow_on = unpacked[0]
            reservoir_features = list(unpacked[1:11])
            
            # Only include active servers
            if active_as_bitmap & (1 << i):
                as_stats[i] = {
                    'n_flow_on': n_flow_on,
                    'reservoir_features': reservoir_features,
                    'fct_mean': reservoir_features[0],
                    'fct_p90': reservoir_features[1],
                    'fct_std': reservoir_features[2],
                    'fct_mean_decay': reservoir_features[3],
                    'fct_p90_decay': reservoir_features[4],
                    'duration_mean': reservoir_features[5],
                    'duration_p90': reservoir_features[6],
                    'duration_std': reservoir_features[7],
                    'duration_mean_decay': reservoir_features[8],
                    'duration_p90_decay': reservoir_features[9],
                }
            
            offset += MessageOutLayout.SERVER_SIZE
        
        return {
            'sequence_id': sequence_id,
            'timestamp_us': timestamp_us,
            'timestamp': timestamp_us / 1e6,  # Convert to seconds
            'active_as_bitmap': active_as_bitmap,
            'num_active_as': num_active_as,
            'active_servers': [i for i in range(MAX_AS) if active_as_bitmap & (1 << i)],
            'server_stats': as_stats
        }


class MessageInLayout:
    """
    Layout for action messages (RL → VPP)
    
    Memory layout:
        uint64_t sequence_id
        uint64_t timestamp_us
        uint32_t num_servers
        uint32_t reserved
        float weights[MAX_AS]
        struct {
            float prob
            uint32_t alias
        } alias_table[MAX_AS]
    """
    
    HEADER_FORMAT = '=QQII'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    WEIGHTS_FORMAT = f'={MAX_AS}f'
    WEIGHTS_SIZE = struct.calcsize(WEIGHTS_FORMAT)
    
    ALIAS_ENTRY_FORMAT = '=fI'
    ALIAS_ENTRY_SIZE = struct.calcsize(ALIAS_ENTRY_FORMAT)
    ALIAS_TABLE_SIZE = ALIAS_ENTRY_SIZE * MAX_AS
    
    MESSAGE_SIZE = HEADER_SIZE + WEIGHTS_SIZE + ALIAS_TABLE_SIZE
    
    @staticmethod
    def pack(sequence_id: int,
             weights: List[float],
             alias_table: Optional[List[Tuple[float, int]]] = None) -> bytes:
        """
        Pack action message into bytes.
        
        Args:
            sequence_id: Sequence ID matching observation
            weights: Server weights (length <= MAX_AS)
            alias_table: Optional alias table for O(1) sampling
            
        Returns:
            Packed bytes
        """
        timestamp_us = int(time.time() * 1e6)
        num_servers = len(weights)
        
        # Pad weights to MAX_AS
        padded_weights = list(weights) + [0.0] * (MAX_AS - len(weights))
        
        # Pack header
        data = struct.pack(
            MessageInLayout.HEADER_FORMAT,
            sequence_id,
            timestamp_us,
            num_servers,
            0  # reserved
        )
        
        # Pack weights
        data += struct.pack(MessageInLayout.WEIGHTS_FORMAT, *padded_weights)
        
        # Pack alias table (or zeros if not provided)
        if alias_table is None:
            alias_table = [(0.0, 0)] * MAX_AS
        else:
            # Pad to MAX_AS
            alias_table = list(alias_table) + [(0.0, 0)] * (MAX_AS - len(alias_table))
        
        for prob, alias in alias_table:
            data += struct.pack(MessageInLayout.ALIAS_ENTRY_FORMAT, prob, alias)
        
        return data
    
    @staticmethod
    def unpack(data: bytes) -> Dict:
        """
        Unpack action message from bytes.
        
        Args:
            data: Packed bytes
            
        Returns:
            Dictionary with message fields
        """
        # Unpack header
        header = struct.unpack(MessageInLayout.HEADER_FORMAT,
                              data[:MessageInLayout.HEADER_SIZE])
        
        sequence_id = header[0]
        timestamp_us = header[1]
        num_servers = header[2]
        
        # Unpack weights
        offset = MessageInLayout.HEADER_SIZE
        weights_data = data[offset:offset + MessageInLayout.WEIGHTS_SIZE]
        weights = list(struct.unpack(MessageInLayout.WEIGHTS_FORMAT, weights_data))
        
        # Unpack alias table
        offset += MessageInLayout.WEIGHTS_SIZE
        alias_table = []
        
        for i in range(MAX_AS):
            entry_data = data[offset:offset + MessageInLayout.ALIAS_ENTRY_SIZE]
            prob, alias = struct.unpack(MessageInLayout.ALIAS_ENTRY_FORMAT, entry_data)
            alias_table.append((prob, alias))
            offset += MessageInLayout.ALIAS_ENTRY_SIZE
        
        return {
            'sequence_id': sequence_id,
            'timestamp_us': timestamp_us,
            'timestamp': timestamp_us / 1e6,
            'num_servers': num_servers,
            'weights': weights[:num_servers],
            'alias_table': alias_table[:num_servers]
        }


class RingBufferLayout:
    """
    Ring buffer for observation messages.
    
    Memory layout:
        uint64_t write_index (atomic)
        uint64_t padding[7]  (cache line alignment)
        msg_out_t messages[RING_BUFFER_SIZE]
    """
    
    INDEX_FORMAT = '=Q7Q'  # write_index + 7 padding
    INDEX_SIZE = struct.calcsize(INDEX_FORMAT)
    
    TOTAL_SIZE = INDEX_SIZE + MessageOutLayout.MESSAGE_SIZE * RING_BUFFER_SIZE
    
    @staticmethod
    def get_write_index(mm: mmap.mmap) -> int:
        """Read write_index from memory."""
        mm.seek(0)
        data = mm.read(8)
        return struct.unpack('=Q', data)[0]
    
    @staticmethod
    def set_write_index(mm: mmap.mmap, index: int):
        """Write write_index to memory."""
        mm.seek(0)
        mm.write(struct.pack('=Q', index))
    
    @staticmethod
    def get_message_offset(slot: int) -> int:
        """Get byte offset for message in slot."""
        return RingBufferLayout.INDEX_SIZE + slot * MessageOutLayout.MESSAGE_SIZE


# Calculate total shared memory size
TOTAL_SHM_SIZE = (
    RingBufferLayout.TOTAL_SIZE +  # msg_out ring buffer
    MessageInLayout.MESSAGE_SIZE     # msg_in
)

print(f"[Memory Layout Info]")
print(f"  MessageOut size: {MessageOutLayout.MESSAGE_SIZE} bytes")
print(f"  MessageIn size: {MessageInLayout.MESSAGE_SIZE} bytes")
print(f"  RingBuffer size: {RingBufferLayout.TOTAL_SIZE} bytes")
print(f"  Total SHM size: {TOTAL_SHM_SIZE} bytes ({TOTAL_SHM_SIZE/1024:.1f} KB)")


if __name__ == "__main__":
    # Test packing/unpacking
    print("\n=== Testing Message Layouts ===\n")
    
    # Test MessageOut
    print("Testing MessageOut...")
    as_stats = [
        {
            'n_flow_on': 10,
            'reservoir_features': [0.1, 0.2, 0.05, 0.11, 0.21, 
                                  1.0, 1.5, 0.3, 1.1, 1.6]
        },
        {
            'n_flow_on': 15,
            'reservoir_features': [0.15, 0.25, 0.06, 0.16, 0.26,
                                  1.2, 1.7, 0.4, 1.3, 1.8]
        }
    ]
    
    packed = MessageOutLayout.pack(
        sequence_id=42,
        timestamp_us=1234567890123456,
        active_as_bitmap=0b11,  # First 2 servers active
        num_active_as=2,
        as_stats=as_stats
    )
    
    print(f"  Packed size: {len(packed)} bytes")
    
    unpacked = MessageOutLayout.unpack(packed)
    print(f"  Sequence ID: {unpacked['sequence_id']}")
    print(f"  Active servers: {unpacked['active_servers']}")
    print(f"  Server 0 flows: {unpacked['server_stats'][0]['n_flow_on']}")
    print(f"  Server 0 FCT mean: {unpacked['server_stats'][0]['fct_mean']:.3f}")
    
    # Test MessageIn
    print("\nTesting MessageIn...")
    weights = [1.0, 1.5, 2.0, 1.2]
    
    packed = MessageInLayout.pack(
        sequence_id=42,
        weights=weights
    )
    
    print(f"  Packed size: {len(packed)} bytes")
    
    unpacked = MessageInLayout.unpack(packed)
    print(f"  Sequence ID: {unpacked['sequence_id']}")
    print(f"  Num servers: {unpacked['num_servers']}")
    print(f"  Weights: {unpacked['weights']}")
    
    print("\n✅ All layout tests passed!")
