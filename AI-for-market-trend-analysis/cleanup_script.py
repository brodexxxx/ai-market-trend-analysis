#!/usr/bin/env python3
"""
Script to remove all instances of '' from Python files in the project
"""

import os
import glob

def clean_file(filepath):
    """Remove all instances of '' from a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if '' in content:
            original_content = content
            content = content.replace('', '')

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ Cleaned {filepath}")
            return True
        else:
            return False
    except Exception as e:
        print(f"❌ Error cleaning {filepath}: {e}")
        return False

def main():
    """Main cleanup function"""
    print("🧹 Starting cleanup of '' characters from codebase...")

    # Find all Python files
    python_files = glob.glob('*.py') + glob.glob('src/*.py') + glob.glob('ai-market-trends-analyzer/**/*.py', recursive=True)

    cleaned_count = 0
    total_files = len(python_files)

    for filepath in python_files:
        if clean_file(filepath):
            cleaned_count += 1

    print(f"\n🎉 Cleanup complete!")
    print(f"📊 Files processed: {total_files}")
    print(f"🧽 Files cleaned: {cleaned_count}")
    print(f"✨ '' characters removed from {cleaned_count} files")

if __name__ == "__main__":
    main()
