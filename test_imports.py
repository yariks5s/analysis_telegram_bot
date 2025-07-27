#!/usr/bin/env python3
"""
Simple test script to verify that all major modules can be imported
after the restructuring.
"""

print("Testing imports from the restructured project...")

# Test core imports
print("Testing core modules...")
try:
    from src.core import config, utils, preferences
    print("✓ Core modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import core modules: {e}")

# Test database imports
print("\nTesting database modules...")
try:
    from src.database import models, operations
    print("✓ Database modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import database modules: {e}")

# Test API imports
print("\nTesting API modules...")
try:
    from src.api import data_fetcher
    print("✓ API modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import API modules: {e}")

# Test visualization imports
print("\nTesting visualization modules...")
try:
    from src.visualization import chart_styles, plot_builder
    print("✓ Visualization modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import visualization modules: {e}")

# Test model_classes imports
print("\nTesting model_classes modules...")
try:
    from src.model_classes import indicators
    print("✓ Model classes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import model classes: {e}")

# Test telegram imports
print("\nTesting telegram modules...")
try:
    from src.telegram import handlers
    from src.telegram.signals import detection
    print("✓ Telegram modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import telegram modules: {e}")

# Test analysis modules
print("\nTesting analysis modules...")
try:
    from src.analysis.utils import helpers
    print("✓ Analysis modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import analysis modules: {e}")

print("\nAll import tests completed!")
