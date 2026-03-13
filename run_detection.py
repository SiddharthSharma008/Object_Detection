#!/usr/bin/env python3
"""
Quick Start: Gym Equipment Detection
Just run this to detect equipment in default locations
"""

from detect_gym_equipment import *
import sys

def quick_detect(source='GymFrames', output='DetectedEquipment', confidence=0.25):
    """Quick detection with custom parameters."""
    global INPUT_SOURCE, OUTPUT_DIR, CONFIDENCE_THRESHOLD
    INPUT_SOURCE = source
    OUTPUT_DIR = output
    CONFIDENCE_THRESHOLD = confidence
    main()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Gym Equipment Detection')
    parser.add_argument('-s', '--source', type=str, default='GymFrames', 
                        help='Input source (file or directory)')
    parser.add_argument('-o', '--output', type=str, default='DetectedEquipment',
                        help='Output directory')
    parser.add_argument('-c', '--confidence', type=float, default=0.25,
                        help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    quick_detect(args.source, args.output, args.confidence)
