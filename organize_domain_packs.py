"""
Utility Script: Organize Industry JSON Files into Domain Packs Structure
=========================================================================

This script organizes your existing industry-specific JSON files into
the proper domain_packs folder structure for the Dynamic NLP Pipeline.

Usage:
    python organize_domain_packs.py

Directory Structure Created:
    domain_packs/
    ├── Banking/
    │   ├── keywords.json
    │   └── rules.json
    ├── E-commerce/
    │   ├── keywords.json
    │   └── rules.json
    └── ... (other industries)
"""

import json
import os
import shutil
from pathlib import Path

# Industry name mappings (filename prefix -> proper industry name)
INDUSTRY_MAPPINGS = {
    'banking': 'Banking',
    'ecommerce': 'E-commerce',
    'e-commerce': 'E-commerce',
    'financial_services': 'Financial_Services',
    'healthcare': 'Healthcare',
    'streaming_entertainment': 'Streaming_Entertainment',
    'technology_software': 'Technology_Software',
    'telecommunications': 'Telecommunications',
    'transportation': 'Transportation',
    'travel_hospitality': 'Travel_Hospitality'
}


def organize_domain_packs(source_dir: str, output_dir: str = 'domain_packs'):
    """
    Organize industry JSON files into domain_packs structure
    
    Args:
        source_dir: Directory containing the JSON files
        output_dir: Output directory for domain_packs
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"📁 Creating domain packs structure in: {output_path.absolute()}")
    print("="*70)
    
    # Get all JSON files in source directory
    source_path = Path(source_dir)
    json_files = list(source_path.glob('*.json'))
    
    # Separate files by type
    rules_files = {}
    keywords_files = {}
    other_files = []
    
    for file in json_files:
        filename = file.stem.lower()
        
        # Skip system files
        if 'system' in filename or 'master' in filename or 'mapping' in filename:
            other_files.append(file)
            continue
        
        # Determine industry name
        industry = None
        for prefix, proper_name in INDUSTRY_MAPPINGS.items():
            if filename.startswith(prefix):
                industry = proper_name
                break
        
        if not industry:
            print(f"⚠️  Warning: Could not determine industry for {file.name}")
            other_files.append(file)
            continue
        
        # Categorize by type
        if 'rule' in filename:
            rules_files[industry] = file
        elif 'keyword' in filename:
            keywords_files[industry] = file
        else:
            print(f"⚠️  Warning: Unknown file type for {file.name}")
            other_files.append(file)
    
    # Create industry directories and copy files
    industries_created = set()
    
    for industry in rules_files.keys():
        industry_dir = output_path / industry
        industry_dir.mkdir(exist_ok=True)
        
        # Copy rules file
        if industry in rules_files:
            rules_dest = industry_dir / 'rules.json'
            shutil.copy2(rules_files[industry], rules_dest)
            print(f"✓ {industry}/rules.json")
        
        # Copy keywords file
        if industry in keywords_files:
            keywords_dest = industry_dir / 'keywords.json'
            shutil.copy2(keywords_files[industry], keywords_dest)
            print(f"✓ {industry}/keywords.json")
        
        industries_created.add(industry)
    
    # Copy company mapping to root
    company_mapping_file = None
    for file in other_files:
        if 'mapping' in file.stem.lower():
            company_mapping_file = file
            break
    
    if company_mapping_file:
        mapping_dest = output_path / 'company_industry_mapping.json'
        shutil.copy2(company_mapping_file, mapping_dest)
        print(f"\n✓ company_industry_mapping.json (copied to root)")
    
    # Summary
    print("\n" + "="*70)
    print(f"✅ Successfully organized {len(industries_created)} industries!")
    print(f"\nIndustries created:")
    for industry in sorted(industries_created):
        print(f"  • {industry}")
    
    print(f"\n📂 Domain packs location: {output_path.absolute()}")
    
    # Create README
    create_readme(output_path, industries_created)
    
    print("\n✅ Setup complete! You can now use these files with the Dynamic NLP Pipeline.")


def create_readme(output_path: Path, industries: set):
    """Create a README file explaining the structure"""
    
    readme_content = f"""# Domain Packs - Industry-Specific Rules & Keywords

This directory contains industry-specific classification rules and keywords
for the Dynamic NLP Pipeline.

## Directory Structure

```
domain_packs/
├── company_industry_mapping.json  # Maps companies to industries
{''.join([f"├── {ind}/" + chr(10) + f"│   ├── keywords.json" + chr(10) + f"│   └── rules.json" + chr(10) for ind in sorted(industries)])}
└── README.md  # This file
```

## Industries Available

{chr(10).join([f"- **{ind}**" for ind in sorted(industries)])}

## File Format

### keywords.json
Contains keyword groups for quick categorization:
```json
[
  {{
    "conditions": ["keyword1", "keyword2", "keyword3"],
    "set": {{
      "category": "Main Category",
      "subcategory": "Subcategory",
      "sentiment": "positive|negative|neutral"
    }}
  }}
]
```

### rules.json
Contains detailed classification rules:
```json
[
  {{
    "conditions": ["condition1", "condition2"],
    "set": {{
      "category": "L1 Category",
      "subcategory": "L2 Subcategory",
      "level_3": "L3 Tertiary",
      "level_4": "L4 Quaternary",
      "sentiment": "positive|negative|neutral"
    }}
  }}
]
```

## Usage in Streamlit App

1. Open the Dynamic NLP Pipeline app
2. In the sidebar, upload:
   - `rules.json` for your industry
   - `keywords.json` for your industry
   - `company_industry_mapping.json` (optional)
3. Enter the industry name
4. Click "Load Industry Configuration"
5. Select the loaded industry from the dropdown
6. Upload your data and run analysis!

## Adding New Industries

To add a new industry:

1. Create a new directory: `domain_packs/Your_Industry/`
2. Add `keywords.json` with keyword rules
3. Add `rules.json` with classification rules
4. Update `company_industry_mapping.json` if needed
5. Load in the Streamlit app

## Statistics

Total industries: {len(industries)}
Last updated: {Path(__file__).stat().st_mtime if Path(__file__).exists() else 'N/A'}
"""
    
    readme_path = output_path / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Created README.md")


def validate_json_files(domain_packs_dir: str = 'domain_packs'):
    """Validate all JSON files in domain packs"""
    
    print("\n" + "="*70)
    print("🔍 Validating JSON files...")
    print("="*70)
    
    domain_path = Path(domain_packs_dir)
    
    if not domain_path.exists():
        print(f"❌ Directory not found: {domain_path}")
        return False
    
    all_valid = True
    
    for industry_dir in domain_path.iterdir():
        if not industry_dir.is_dir():
            continue
        
        industry_name = industry_dir.name
        print(f"\n📁 {industry_name}/")
        
        # Check rules.json
        rules_file = industry_dir / 'rules.json'
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    rules = json.load(f)
                print(f"  ✓ rules.json: {len(rules)} rules")
            except Exception as e:
                print(f"  ❌ rules.json: Invalid JSON - {e}")
                all_valid = False
        else:
            print(f"  ⚠️  rules.json: Not found")
            all_valid = False
        
        # Check keywords.json
        keywords_file = industry_dir / 'keywords.json'
        if keywords_file.exists():
            try:
                with open(keywords_file, 'r') as f:
                    keywords = json.load(f)
                print(f"  ✓ keywords.json: {len(keywords)} keyword groups")
            except Exception as e:
                print(f"  ❌ keywords.json: Invalid JSON - {e}")
                all_valid = False
        else:
            print(f"  ⚠️  keywords.json: Not found")
            all_valid = False
    
    print("\n" + "="*70)
    if all_valid:
        print("✅ All JSON files are valid!")
    else:
        print("❌ Some files have issues. Please review above.")
    
    return all_valid


if __name__ == "__main__":
    import sys
    
    # Get source directory from command line or use default
    if len(sys.argv) > 1:
        source_directory = sys.argv[1]
    else:
        # Default to current directory
        source_directory = "."
    
    print("🚀 Domain Packs Organizer")
    print("="*70)
    print(f"Source directory: {Path(source_directory).absolute()}")
    print()
    
    # Organize files
    try:
        organize_domain_packs(source_directory)
        
        # Validate
        print()
        validate_json_files()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
