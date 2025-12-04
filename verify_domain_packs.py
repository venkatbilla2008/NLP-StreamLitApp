#!/usr/bin/env python3
"""
Domain Packs Structure Verification Script
Run this in your repo root to verify all industries are properly set up
"""

import os
import json
import sys

def check_domain_packs():
    """Check domain_packs directory structure"""
    
    print("=" * 70)
    print("DOMAIN PACKS STRUCTURE VERIFICATION")
    print("=" * 70)
    
    domain_dir = "domain_packs"
    
    # Check if directory exists
    if not os.path.exists(domain_dir):
        print(f"❌ ERROR: Directory not found: {domain_dir}")
        print(f"   Current directory: {os.getcwd()}")
        return False
    
    print(f"✅ Directory exists: {domain_dir}\n")
    
    # Expected industries
    expected_industries = [
        "Banking",
        "E-commerce",
        "Financial_Services",
        "Healthcare",
        "Other",
        "Streaming_Entertainment",
        "Technology_Software",
        "Telecommunications",
        "Transportation",
        "Travel_Hospitality"
    ]
    
    print(f"Expected industries: {len(expected_industries)}")
    print(f"  {', '.join(expected_industries)}\n")
    
    # Scan directory
    try:
        items = os.listdir(domain_dir)
        print(f"Found {len(items)} items in {domain_dir}:\n")
    except Exception as e:
        print(f"❌ ERROR reading directory: {e}")
        return False
    
    # Check each item
    valid_industries = []
    issues_found = []
    
    for item in sorted(items):
        item_path = os.path.join(domain_dir, item)
        
        # Skip non-directories
        if not os.path.isdir(item_path):
            print(f"⚠️  {item:30} - Not a directory (skipped)")
            continue
        
        # Skip hidden directories
        if item.startswith('.'):
            print(f"⚠️  {item:30} - Hidden directory (skipped)")
            continue
        
        # Check for required files
        rules_path = os.path.join(item_path, "rules.json")
        keywords_path = os.path.join(item_path, "keywords.json")
        
        has_rules = os.path.exists(rules_path)
        has_keywords = os.path.exists(keywords_path)
        
        if has_rules and has_keywords:
            # Verify JSON files are valid
            try:
                with open(rules_path, 'r') as f:
                    rules = json.load(f)
                with open(keywords_path, 'r') as f:
                    keywords = json.load(f)
                
                print(f"✅ {item:30} - {len(rules)} rules, {len(keywords)} keywords")
                valid_industries.append(item)
            
            except json.JSONDecodeError as e:
                print(f"❌ {item:30} - Invalid JSON: {e}")
                issues_found.append(f"{item}: Invalid JSON")
            
            except Exception as e:
                print(f"❌ {item:30} - Error: {e}")
                issues_found.append(f"{item}: {e}")
        
        else:
            missing = []
            if not has_rules:
                missing.append("rules.json")
            if not has_keywords:
                missing.append("keywords.json")
            
            print(f"❌ {item:30} - Missing: {', '.join(missing)}")
            issues_found.append(f"{item}: Missing {', '.join(missing)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Valid industries: {len(valid_industries)}/{len(expected_industries)}")
    if valid_industries:
        print(f"   {', '.join(sorted(valid_industries))}")
    
    # Check for missing expected industries
    missing_expected = set(expected_industries) - set(valid_industries)
    if missing_expected:
        print(f"\n❌ Missing expected industries: {len(missing_expected)}")
        for ind in sorted(missing_expected):
            print(f"   - {ind}")
    
    # Show issues
    if issues_found:
        print(f"\n⚠️  Issues found: {len(issues_found)}")
        for issue in issues_found:
            print(f"   - {issue}")
    
    # Check company mapping
    print("\n" + "-" * 70)
    mapping_path = os.path.join(domain_dir, "company_industry_mapping.json")
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            industries_in_mapping = len(mapping.get('industries', {}))
            print(f"✅ Company mapping exists: {industries_in_mapping} industries mapped")
        except Exception as e:
            print(f"⚠️  Company mapping exists but has errors: {e}")
    else:
        print(f"⚠️  Company mapping not found (optional)")
    
    print("\n" + "=" * 70)
    
    if len(valid_industries) == len(expected_industries):
        print("✅ SUCCESS: All industries are properly configured!")
        return True
    else:
        print(f"⚠️  WARNING: Only {len(valid_industries)}/{len(expected_industries)} industries configured")
        print("\nTO FIX:")
        print("1. Ensure all industry folders exist in domain_packs/")
        print("2. Each folder must contain both rules.json and keywords.json")
        print("3. JSON files must be valid (no syntax errors)")
        return False

if __name__ == "__main__":
    success = check_domain_packs()
    sys.exit(0 if success else 1)
