#!/usr/bin/env python3
"""
Script to fix imports in test files after project restructuring.

This script updates old module paths to the new module structure.
"""

import os
import re
import glob

# Define the import mappings (old import -> new import)
IMPORT_MAPPINGS = {
    "message_handlers": "src.telegram.handlers",
    "indicators": "src.analysis.detection.indicators", 
    "helpers": "src.analysis.utils.helpers",
    "preferences": "src.core.preferences",
    "plot_build_helpers": "src.visualization.plot_builder",
    "data_fetching_instruments": "src.api.data_fetcher",
    "database": "src.database.operations",
}

# Module paths for specific classes and utilities
CLASS_MAPPINGS = {
    "IndicatorUtils.breaker_block_utils": "src.analysis.utils.breaker_block_utils",
    "IndicatorUtils.fvg_utils": "src.analysis.utils.fvg_utils",
    "IndicatorUtils.liquidity_level_utils": "src.analysis.utils.liquidity_level_utils",
    "IndicatorUtils.order_block_utils": "src.analysis.utils.order_block_utils",
    "IndicatorUtils.liquidity_pool_utils": "src.analysis.utils.liquidity_pool_utils"
}

# Path to test directory
TEST_DIR = "tests"

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    print(f"Processing {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading file: {e}")
        return
    
    # Keep track of changes
    original_content = content
    changes_made = False
    
    # Replace standard module imports
    for old_import, new_import in IMPORT_MAPPINGS.items():
        # Match both 'import old_import' and 'from old_import import ...'
        old_content = content
        content = re.sub(
            rf'(from|import)\s+{old_import}([^\w\.]|$)', 
            rf'\1 {new_import}\2', 
            content
        )
        
        # Also handle "mocker.patch("old_import.something")" cases
        content = content.replace(f'"{old_import}.', f'"{new_import}.')
        
        if content != old_content:
            changes_made = True
            print(f"  Updated '{old_import}' to '{new_import}'")
    
    # Replace class imports from IndicatorUtils
    for old_class_path, new_class_path in CLASS_MAPPINGS.items():
        old_content = content
        
        # Handle from IndicatorUtils.x import y pattern
        content = re.sub(
            rf'from\s+{old_class_path}\s+import', 
            f'from {new_class_path} import', 
            content
        )
        
        # Handle import IndicatorUtils.x pattern
        content = re.sub(
            rf'import\s+{old_class_path}([^\w\.]|$)', 
            f'import {new_class_path}\1', 
            content
        )
        
        # Handle mocker.patch cases
        content = content.replace(f'"{old_class_path}.', f'"{new_class_path}.')
        
        if content != old_content:
            changes_made = True
            print(f"  Updated '{old_class_path}' to '{new_class_path}'")
    
    # Save changes if the content was modified
    if changes_made:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Successfully updated imports in {file_path}")
        except Exception as e:
            print(f"  Error writing file: {e}")
    else:
        print(f"  No changes made to {file_path}")

def main():
    """Update imports in all Python test files."""
    test_files = glob.glob(os.path.join(TEST_DIR, "*.py"))
    
    for file_path in test_files:
        fix_imports_in_file(file_path)
    
    print("Import fixing complete!")

if __name__ == "__main__":
    main()
