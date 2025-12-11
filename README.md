
def retrieve_relevant_context(vector_store, raw_data_store, query, n_results=15):
    """Retrieve relevant context from vector database"""
    try:
        results = query_faiss(vector_store, query, n_results=n_results)
        
        context_blocks = []
        if results['metadatas']:
            for i, metadata in enumerate(results['metadatas']):
                chunk_id = metadata.get('chunk_id')
                if chunk_id in raw_data_store:
                    context_blocks.append(raw_data_store[chunk_id])
                elif i < len(results['documents']):
                    # If chunk not in raw_data_store, create a new entry
                    context_blocks.append({
                        "content": results['documents'][i],
                        "page": metadata.get('page_number', '?'),
                        "type": "text"
                    })
        
        # If no results, get some general content
        if not context_blocks:
            for k, v in list(raw_data_store.items())[:n_results]:
                if v.get('type') == 'text':
                    context_blocks.append(v)
        
        return context_blocks
    except Exception as e:
        print(f"⚠️ Context retrieval failed: {e}")
        return []

def find_table_data_in_raw_store(raw_data_store):
    """Find table-like data in raw_data_store"""
    print("🔍 Looking for table data in raw store...")
    
    table_blocks = []
    
    for key, value in raw_data_store.items():
        if value.get('type') == 'text':
            content = value.get('content', '')
            page = value.get('page', '?')
            
            # Check for table patterns
            lines = content.split('\n')
            
            # Look for table indicators
            table_indicators = [
                # Service type keywords
                'drugs.*biological',
                'laboratory.*pathology',
                'dme.*supplies.*orthotics',
                'therapy services',
                'physician extender',
                'all other services',
                'unspecified services',
                
                # Fee schedule codes
                '005-460',
                '201-544',
                '201-518',
                
                # Line of business
                'commercial.*ppo',
                'medicare.*advantage',
                'medicaid.*pathways'
            ]
            
            # Check if content has multiple table indicators
            indicator_count = 0
            content_lower = content.lower()
            for indicator in table_indicators:
                if re.search(indicator, content_lower):
                    indicator_count += 1
            
            # Also check for tabular structure
            has_pipe_structure = any('|' in line for line in lines[:10])
            has_tab_separators = any('\t' in line for line in lines[:10])
            
            if indicator_count >= 2 or has_pipe_structure or has_tab_separators:
                # Also check if it has rates
                if '%' in content:
                    table_blocks.append(value)
                    print(f"   📊 Found table-like data on page {page}")
    
    print(f"   Total table blocks found: {len(table_blocks)}")
    return table_blocks

def retrieve_comprehensive_context(vector_store, raw_data_store, max_chunks=35):
    """Get context with TABLES FIRST, then rates, then contract info"""
    print(f"🔍 Retrieving context (TABLES PRIORITY)...")
    
    all_blocks = []
    
    # Step 1: Get TABLE data first (MOST IMPORTANT)
    print("📊 Looking for table data...")
    table_blocks = find_table_data_in_raw_store(raw_data_store)
    
    if table_blocks:
        all_blocks.extend(table_blocks)
        print(f"   Added {len(table_blocks)} table blocks")
    else:
        print("   ⚠️ No table blocks found, will search differently")
    
    # Step 2: Get pages with fee schedule codes
    print("💰 Looking for fee schedule pages...")
    fee_schedule_pages = []
    for key, value in raw_data_store.items():
        if value.get('type') == 'text':
            content = value.get('content', '')
            if any(code in content for code in ['005-460', '201-544', '201-518']):
                if value not in all_blocks:
                    fee_schedule_pages.append(value)
    
    if fee_schedule_pages:
        all_blocks.extend(fee_schedule_pages[:5])
        print(f"   Added {len(fee_schedule_pages)} fee schedule pages")
    
    # Step 3: Get rate pages
    print("📈 Looking for rate pages...")
    rate_pages = []
    for key, value in raw_data_store.items():
        if value.get('type') == 'text':
            content = value.get('content', '')
            page = value.get('page', 0)
            
            # More specific rate detection
            if re.search(r'\b(100|45|75|85|80|72)%\b', content):
                if value not in all_blocks:
                    rate_pages.append(value)
    
    if rate_pages:
        all_blocks.extend(rate_pages[:10])
        print(f"   Added {len(rate_pages)} rate pages")
    
    # Step 4: Get contract identification pages
    print("📄 Looking for contract pages...")
    contract_pages = []
    for key, value in raw_data_store.items():
        if value.get('type') == 'text':
            content = value.get('content', '').lower()
            page = value.get('page', 0)
            
            contract_keywords = [
                'contract no',
                'agreement id',
                'provider name',
                'npi',
                'tax id',
                'effective date',
                'city of fishers'
            ]
            
            if any(keyword in content for keyword in contract_keywords):
                if value not in all_blocks and page <= 10:
                    contract_pages.append(value)
    
    if contract_pages:
        all_blocks.extend(contract_pages[:5])
        print(f"   Added {len(contract_pages)} contract pages")
    
    # Step 5: If we still need more content, add pages with service types
    if len(all_blocks) < max_chunks:
        print("➕ Adding more content pages...")
        for key, value in raw_data_store.items():
            if value.get('type') == 'text' and value not in all_blocks:
                content = value.get('content', '').lower()
                
                # Add pages with service-related content
                service_keywords = [
                    'service type',
                    'line of business',
                    'reimbursement',
                    'fee schedule'
                ]
                
                if any(keyword in content for keyword in service_keywords):
                    all_blocks.append(value)
                    if len(all_blocks) >= max_chunks:
                        break
    
    # Remove duplicates
    unique_blocks = []
    seen_content = set()
    
    for block in all_blocks:
        content = block.get('content', '')
        content_key = content[:500].strip()
        
        if content_key and content_key not in seen_content:
            seen_content.add(content_key)
            unique_blocks.append(block)
    
    # Take up to max_chunks
    final_blocks = unique_blocks[:max_chunks]
    
    # DEBUG: Show what we're getting
    print(f"\n📚 FINAL Context Selection ({len(final_blocks)} blocks):")
    for i, block in enumerate(final_blocks[:10]):
        page = block.get('page', '?')
        content_preview = block.get('content', '')[:80].replace('\n', ' ')
        
        # Classify
        block_type = "📄"
        content_lower = block.get('content', '').lower()
        if any(code in content_lower for code in ['005-460', '201-544', '201-518']):
            block_type = "💰"
        elif any(term in content_lower for term in ['drugs', 'laboratory', 'dme', 'therapy']):
            block_type = "📊"
        
        print(f"  {i+1:2d}. Page {page:2d} {block_type} {content_preview}...")
    
    return final_blocks

# ============================================================================
# SINGLE-CALL EXTRACTION FUNCTION (STRICT - NO ASSUMPTIONS)
# ============================================================================

SYSTEM_PROMPT_HEALTHCARE = """
You are a healthcare contract data extraction specialist. Your ONLY task is to extract EXACT values from contract text.

# CRITICAL RULES YOU MUST FOLLOW:

## 1. EXTRACTION RULES:
- Extract ONLY values EXPLICITLY written in the text
- Copy values EXACTLY as written - no modifications
- If information is NOT in the text, output "N/A"
- NEVER invent, assume, or guess values
- NEVER use placeholder examples (no 1234567890, no 2024-01-01, no 99213)

## 2. DATA HANDLING RULES:
- For NPI: Only extract 10-digit numbers explicitly labeled as NPI
- For Tax ID: Only extract numbers explicitly labeled as Tax ID, EIN, or Federal Tax ID
- For Dates: Only extract dates explicitly labeled with date terms
- For Rates: Only extract percentages ($X%, X%) or dollar amounts explicitly mentioned
- For Codes: Only extract CPT/HCPCS codes if explicitly listed

## 3. OUTPUT RULES:
- Always return valid JSON
- Use the exact field names specified in the user prompt
- Maintain consistent JSON structure
- No explanations, no additional text

## 4. VIOLATION PENALTIES:
You will be penalized for:
- Inventing fake NPI numbers
- Inventing fake dates
- Inventing fake codes
- Using example values
- Making assumptions
- Deviating from the JSON structure

Your performance is measured by EXACT extraction accuracy.
"""

USER_PROMPT_HEALTHCARE = """
Extract healthcare contract data from the text below. Extract ALL fields mentioned.

# ALL FIELDS TO EXTRACT FROM TEXT:

## 1. CONTRACT IDENTIFICATION (look on COVER SHEET and TERM sections):
- Contract/Agreement Number (look for: Icertis Contract Number, Contract No.)
- Provider Name (look for: Provider Name, Legal Name, DBA Name)
- NPI (10-digit number labeled NPI)
- Tax ID / EIN (look for: Federal Tax ID, EIN)
- ZIP Code (look for: Zip:)
- Provider Specialty (look for: Provider Specialty:)
- Effective From Date (look for: Creation Date, Effective Date)
- Effective To Date (look for: "initial term of this Agreement shall be for", "auto-renew")
- Place of Service (look for state abbreviations like IN, TX, CA)

## 2. SERVICE INFORMATION (from FEE SCHEDULE TABLES):

For EACH service type found in tables, extract:
- Line of Business: "Commercial" or "Medicare Advantage" or "Medicaid (PathWays)"
- Service Type: e.g., "Drugs & Biologicals", "Laboratory & Pathology", "DME/Supplies/Orthotics", "Therapy Services", "Physician Extender Services", "All Other Services", "Unspecified Services"
- Service Description: Full description from text
- Services: Short service name
- Age Group: Usually "All" if not specified
- Codes: CPT/HCPCS codes if mentioned (e.g., "97001-98943, 92506-92508")
- Price Rate: Percentage value (e.g., "100%", "45%", "75%", "85%", "80%", "72%")
- Reimbursement Rate: Full reimbursement description (e.g., "100% of Humana's 201-544 fee schedule")
- Reimbursement Methodology: Derived from fee schedule code:
  * 201-544 → "Fee Schedule (Percentage of ASP/industry standard)"
  * 201-518 → "Fee Schedule (Modified 2006 Medicare RBRVS)"
  * 005-460 → "Fee Schedule (Percentage of Medicare RBRVS)"
  * Medicaid → "Indiana Medicaid Fee Schedule (FSSA payment systems)"
- Health Benefit Plans: From PRODUCT PARTICIPATION LIST table with checkmarks (✓ or X)

## 3. LOOK FOR THESE SPECIFIC PATTERNS:
- "Creation Date: 2/22/2024" → effective_from_date: "2/22/2024"
- "initial term of this Agreement shall be for Three (3) year(s)" → effective_to_date: "Auto-renew 3 years"
- "State: Indiana" → placeofserv: "IN"
- "Provider Specialty: APHW-Public Health or Welfare Agency" → provider_speciality: "Public Health or Welfare Agency"
- "Drugs & Biologicals 100% of Humana's 201-544" → service_type: "Drugs & Biologicals", price_rate: "100%"
- "Therapy Codes: 97001 - 98943, 92506 - 92508" → codes: "97001-98943, 92506-92508"
- "Commercial PPO Plans X" in table → include "Commercial PPO Plans" in health_benefit_plans

CONTRACT TEXT:
{context_text}

# OUTPUT FORMAT - MUST EXTRACT ALL FIELDS:
{{
  "contract_info": {{
    "contract_number": "extracted value or N/A",
    "provider_name": "extracted value or N/A", 
    "npi": "extracted value or N/A",
    "tax_id": "extracted value or N/A",
    "zip_code": "extracted value or N/A",
    "provider_specialty": "extracted value or N/A",
    "effective_from_date": "extracted value or N/A",
    "effective_to_date": "extracted value or N/A",
    "place_of_service": "extracted value or N/A"
  }},
  
  "service_rows": [
    {{
      "line_of_business": "Commercial or Medicare Advantage or Medicaid (PathWays)",
      "service_type": "Drugs & Biologicals or Laboratory & Pathology etc.",
      "service_desc": "Full description from text",
      "services": "Short service name",
      "age_group": "All or extracted value",
      "codes": "CPT/HCPCS codes if mentioned or N/A",
      "grouper": "N/A",
      "price_rate": "100% or 45% or 75% etc.",
      "reimbursement_rate": "100% of Humana's 201-544 fee schedule etc.",
      "reimbursement_methodology": "Fee Schedule (Percentage of ASP/industry standard) etc.",
      "health_benefit_plans": "List of plans with checkmarks from table"
    }}
  ]
}}

# IMPORTANT EXAMPLES:
- If you see: "Drugs & Biologicals 100% of Humana's 201-544 fee schedule" → 
  service_type: "Drugs & Biologicals", price_rate: "100%", reimbursement_rate: "100% of Humana's 201-544 fee schedule"

- If you see: "Therapy Codes: 97001 - 98943, 92506 - 92508" → 
  codes: "97001-98943, 92506-92508"

- If you see: "initial term of this Agreement shall be for Three (3) year(s)" → 
  effective_to_date: "Auto-renew 3 years"

- If you see table with "Commercial PPO Plans X" → 
  health_benefit_plans: "Commercial PPO Plans, ..."
"""

# Then update the MAPPING in extract_all_fields_strict:
# Add these mappings:

mappings = {
    # Contract info
    "contract_number": "CIS CTRCT ID",
    "provider_name": "PROVIDER NAME", 
    "npi": "NPI",
    "tax_id": "TAX ID", 
    "zip_code": "PROVZIPCODE",
    "provider_specialty": "PROVIDER SPECIALITY",
    "effective_from_date": "EFFECTIVE FROM DATE",
    "effective_to_date": "EFFECTIVE TO DATE",
    "place_of_service": "PLACEOFSERV",
    
    # Service rows  
    "line_of_business": "LOB IND",
    "service_type": "SERVICE TYPE",
    "service_desc": "SERVICE DESC",
    "services": "SERVICES",
    "age_group": "AGE GROUP",
    "codes": "CODES",
    "grouper": "GROUPER",
    "price_rate": "PRICE RATE",
    "reimbursement_rate": "REIMBURSEMENT RATE",
    "reimbursement_methodology": "REIMBURSEMENT METHODOLOGY",
    "health_benefit_plans": "HEALTH BENEFIT PLANS"
}

def debug_context_building(context_blocks):
    """Debug function to see what's actually in context_blocks"""
    print("\n🔍 DEBUG: Context Blocks Analysis")
    print(f"Total blocks: {len(context_blocks)}")
    
    total_chars = 0
    for i, block in enumerate(context_blocks):
        page = block.get('page', '?')
        content = block.get('content', '')
        content_length = len(content)
        total_chars += content_length
        
        # Show first 50 chars
        preview = content[:50].replace('\n', ' ')
        
        print(f"\nBlock {i+1}: Page {page}, Length: {content_length} chars")
        print(f"Preview: {preview}...")
        
        # Check if content is actually there
        if content_length < 10:
            print(f"⚠️  WARNING: Block {i+1} has very little content!")
    
    print(f"\n📊 Total characters across all blocks: {total_chars}")
    return total_chars

def extract_all_fields_strict(vector_store, raw_data_store, file_name, file_extension):
    print("\n🚀 Stage: Extraction with SYSTEM + USER prompts")
    
    # USE THE PROPER CONTEXT RETRIEVAL FUNCTION
    context_blocks = retrieve_comprehensive_context(vector_store, raw_data_store, max_chunks=25)
    
    # 🔥 NEW: Debug context building
    debug_context_building(context_blocks)
    
    # 🔥 FIXED: Build context text properly
    context_parts = []
    chars_per_page = 3000  # Increased from 800/1500
    max_total_chars = 35000  # Increased limit
    
    total_chars = 0
    for i, ctx in enumerate(context_blocks):
        page = ctx.get('page', '?')
        content = ctx.get('content', '')
        
        if content and content.strip():
            # Clean content
            clean_content = content.strip()
            
            # Calculate how much to take
            if '%' in content:
                # For rate pages, take more
                take_chars = min(chars_per_page * 2, len(clean_content))
            else:
                take_chars = min(chars_per_page, len(clean_content))
            
            # Ensure we don't exceed total limit
            if total_chars + take_chars > max_total_chars:
                take_chars = max_total_chars - total_chars
            
            if take_chars > 100:  # Only add if substantial content
                context_text = clean_content[:take_chars]
                context_parts.append(f"\n{'='*50}\nPAGE {page}\n{'='*50}\n{context_text}")
                total_chars += len(context_text)
                
                print(f"  Added page {page}: {take_chars} chars (total: {total_chars})")
                
                if total_chars >= max_total_chars:
                    print(f"  ⚠️  Reached max context limit of {max_total_chars} chars")
                    break
    
    context_text = "\n".join(context_parts)
    
    print(f"\n📊 Final context: {len(context_parts)} pages, {len(context_text)} chars")
    
    # 🔥 NEW: Save context to file for debugging
    debug_dir = "./debug"
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    context_file = os.path.join(debug_dir, f"context_for_llm_{timestamp}.txt")
    
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write(context_text)
    
    print(f"📄 Context saved to: {context_file}")
    
    # Count rate occurrences
    rate_count = context_text.count('%')
    print(f"💰 Found {rate_count} percentage symbols in context")
    
    if rate_count == 0:
        print("⚠️ WARNING: No rates found in context!")
        # Try to find rate pages directly from raw_data_store
        print("🔍 Searching raw_data_store for rate pages...")
        rate_pages_found = []
        for key, value in raw_data_store.items():
            if value.get('type') == 'text':
                content = value.get('content', '')
                page = value.get('page', '?')
                if '%' in content and page:
                    # Get a bigger chunk
                    clean_content = content.strip()
                    if len(clean_content) > 100:
                        context_parts.append(f"\n{'='*50}\nDIRECT RATE PAGE {page}\n{'='*50}\n{clean_content[:2000]}")
                        rate_pages_found.append(page)
        
        if rate_pages_found:
            print(f"  Added direct rate pages: {rate_pages_found}")
            context_text = "\n".join(context_parts)
    
    # Create user prompt with MORE context
    user_prompt = USER_PROMPT_HEALTHCARE.format(context_text=context_text)
    
    # 🔥 NEW: Save the full prompt for debugging
    prompt_file = os.path.join(debug_dir, f"full_prompt_{timestamp}.txt")
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT_HEALTHCARE[:2000]}...\n\n")
        f.write(f"=== USER PROMPT ===\n{user_prompt[:5000]}...\n\n")
        f.write(f"=== CONTEXT LENGTH ===\n{len(context_text)} chars")
    
    print(f"📄 Full prompt saved to: {prompt_file}")

    try:
        print("🤖 Calling Ollama with SYSTEM + USER prompts...")
        
        response_text = call_ollama_chat(
            system_prompt=SYSTEM_PROMPT_HEALTHCARE,
            user_prompt=user_prompt,
            model=TEXT_LLM_MODEL,
            format="json",
            temperature=0
        )
        
        print(f"✅ LLM response received: {len(response_text)} chars")
        
        # Parse response
        result = json.loads(response_text)
        
        # Initialize all fields as N/A
        contract_fields = {}
        for field in ALL_FIELD_NAMES:
            contract_fields[field] = "N/A"
        
        # Map extracted data to our fields
        if "contract_info" in result:
            ci = result["contract_info"]
            
            # Map ALL contract fields
            contract_mappings = {
                "contract_number": "CIS CTRCT ID",
                "provider_name": "PROVIDER NAME", 
                "npi": "NPI",
                "tax_id": "TAX ID", 
                "zip_code": "PROVZIPCODE",
                "provider_specialty": "PROVIDER SPECIALITY",
                "effective_from_date": "EFFECTIVE FROM DATE",
                "effective_to_date": "EFFECTIVE TO DATE",
                "place_of_service": "PLACEOFSERV"
            }
            
            for source_key, target_key in contract_mappings.items():
                if source_key in ci and ci[source_key] != "N/A":
                    contract_fields[target_key] = safe_string_convert(ci[source_key])
        
        # Extract service rows
        service_rows = []
        if "service_rows" in result and isinstance(result["service_rows"], list):
            for sr in result["service_rows"]:
                if isinstance(sr, dict):
                    row = {}
                    for field in SERVICE_LEVEL_FIELDS:
                        row[field] = "N/A"
                    
                    # Map ALL service fields
                    service_mappings = {
                        "line_of_business": "LOB IND",
                        "service_type": "SERVICE TYPE",
                        "service_desc": "SERVICE DESC",
                        "services": "SERVICES",
                        "age_group": "AGE GROUP",
                        "codes": "CODES",
                        "grouper": "GROUPER",
                        "price_rate": "PRICE RATE",
                        "reimbursement_rate": "REIMBURSEMENT RATE",
                        "reimbursement_methodology": "REIMBURSEMENT METHODOLOGY",
                        "health_benefit_plans": "HEALTH BENEFIT PLANS"
                    }
                    
                    for json_key, field_key in service_mappings.items():
                        if json_key in sr and sr[json_key] != "N/A":
                            row[field_key] = sr[json_key]
                    
                    # Set defaults for fields that are usually the same
                    if row["AGE GROUP"] == "N/A":
                        row["AGE GROUP"] = "All"
                    if row["GROUPER"] == "N/A":
                        row["GROUPER"] = "N/A"
                    
                    service_rows.append(row)        

        print(f"✅ Extracted {len(service_rows)} service rows")

        # Set metadata
        contract_fields["FILE NAME"] = file_name
        contract_fields["FILE EXTENSION"] = file_extension
        contract_fields["NLP USER ID"] = "System"
        contract_fields["NLP PROCESS TIMESTAMP"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"✅ Extracted {len(service_rows)} service rows")
        
        return {
            "success": True,
            "contract_fields": contract_fields,
            "service_rows": service_rows,
            "raw_result": result
        }
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None
 
