# ============================================================================
# DATA PREPARATION AND EXPORT FUNCTIONS
# ============================================================================

def prepare_excel_data_strict(contract_fields, service_rows, file_name):
    """Prepare final data for Excel export - EXACT ORDER, NO ASSUMPTIONS"""
    print("\n📊 Preparing Excel Data (Strict)...")
    
    excel_data = []
    
    if service_rows:
        # Create one row per service
        for service_row in service_rows:
            row_data = {}
            
            # Add ALL fields in EXACT order from ALL_FIELD_NAMES
            for field in ALL_FIELD_NAMES:
                # Get from service row first, then contract fields
                if field in service_row:
                    row_data[field] = service_row[field]
                elif field in contract_fields:
                    row_data[field] = contract_fields[field]
                else:
                    row_data[field] = "N/A"
            
            excel_data.append(row_data)
    else:
        # No service rows, just contract info row
        row_data = {}
        for field in ALL_FIELD_NAMES:
            row_data[field] = contract_fields.get(field, "N/A")
        excel_data.append(row_data)
    
    # Calculate success metrics
    total_fields = len(ALL_FIELD_NAMES)
    extracted_count = 0
    
    for field in ALL_FIELD_NAMES:
        field_has_value = False
        for row in excel_data:
            value = row.get(field, "N/A")
            if value != "N/A" and str(value).strip():
                field_has_value = True
                break
        
        if field_has_value:
            extracted_count += 1
    
    success_rate = (extracted_count / total_fields) * 100 if total_fields > 0 else 0
    
    # Update NLP fields based on actual extraction
    if success_rate > 70:
        contract_fields["NLP ERROR COMMENTS"] = "Success"
    elif success_rate > 40:
        contract_fields["NLP ERROR COMMENTS"] = f"Partial: {extracted_count}/{total_fields} fields"
    else:
        contract_fields["NLP ERROR COMMENTS"] = f"Low extraction: {extracted_count}/{total_fields} fields"
    
    # Update contract fields in all rows
    for row in excel_data:
        row["NLP ERROR COMMENTS"] = contract_fields["NLP ERROR COMMENTS"]
    
    summary = {
        "total_fields": total_fields,
        "extracted_fields": extracted_count,
        "success_rate": f"{success_rate:.1f}%",
        "service_rows": len(excel_data),
        "extraction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": file_name
    }
    
    print(f"✅ Prepared {len(excel_data)} rows for Excel")
    
    return {
        "excel_data": excel_data,
        "summary": summary,
        "contract_fields": contract_fields
    }

def save_to_excel_strict(excel_data, summary, file_name):
    """Save extracted data to Excel file - EXACT COLUMN ORDER"""
    try:
        if not excel_data:
            print("❌ No data to save")
            return None
        
        # Create DataFrame - columns will be in insertion order
        df = pd.DataFrame(excel_data)
        
        # Ensure ALL fields are present in EXACT order
        # Remove any extra columns not in ALL_FIELD_NAMES
        for col in df.columns:
            if col not in ALL_FIELD_NAMES:
                df = df.drop(columns=[col])
        
        # Add missing columns with "N/A"
        for field in ALL_FIELD_NAMES:
            if field not in df.columns:
                df[field] = "N/A"
        
        # Reorder columns to EXACT order
        df = df[ALL_FIELD_NAMES]
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(file_name)[0]
        output_file = f"ccg2_extraction_{base_name}_{timestamp}.xlsx"
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Contract Data', index=False)
            
            # Summary sheet
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Extraction Summary', index=False)
            
            # Adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    column_letter = column[0].column_letter
                    max_length = 0
                    for cell in column:
                        try:
                            cell_value = str(cell.value) if cell.value else ""
                            max_length = max(max_length, len(cell_value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✅ Excel file saved: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Error saving Excel: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_to_json_strict(excel_data, summary, contract_fields, service_rows, file_name):
    """Save full extraction results to JSON"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(file_name)[0]
        output_file = f"ccg2_extraction_{base_name}_{timestamp}.json"
        
        results = {
            "summary": summary,
            "contract_fields": contract_fields,
            "service_rows": service_rows,
            "excel_data": excel_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON file saved: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        return None

# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def run_extraction_pipeline_strict(file_path):
    """Run complete extraction pipeline - STRICT MODE (NO ASSUMPTIONS)"""
    print("🚀 Starting Healthcare Contract Extraction (STRICT MODE)")
    print("=" * 60)
    
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_path)[1].lower().replace(".", "")
    
    print(f"📄 File: {file_name}")
    print(f"🎯 Target Fields: {len(ALL_FIELD_NAMES)}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Stage 1: Process Document
    print("\n▶️ Stage 1: Document Processing")
    doc_result = process_document(file_path)
    if not doc_result["success"]:
        print("❌ Document processing failed")
        return None
    
    chunks = doc_result["chunks"]
    raw_data_store = doc_result["raw_data_store"]
    
    # Stage 2: Store in Vector Database
    print("\n▶️ Stage 2: Vector Storage")
    vector_result = store_in_vector_db(chunks)
    if not vector_result["success"]:
        print("⚠️ Vector storage completed with issues")
        vector_store = None
    else:
        vector_store = vector_result["vector_store"]
    
    # Stage 3: Single-Call Comprehensive Extraction
    print("\n▶️ Stage 3: Comprehensive Extraction (Single LLM Call)")
    extraction_result = extract_all_fields_strict(
        vector_store, 
        raw_data_store, 
        file_name, 
        file_extension
    )
    
    if not extraction_result or not extraction_result["success"]:
        print("❌ Extraction failed")
        return None
    
    contract_fields = extraction_result["contract_fields"]
    service_rows = extraction_result["service_rows"]
    
    # Stage 4: Prepare Excel Data
    print("\n▶️ Stage 4: Excel Preparation")
    preparation_result = prepare_excel_data_strict(
        contract_fields, 
        service_rows, 
        file_name
    )
    
    excel_data = preparation_result["excel_data"]
    summary = preparation_result["summary"]
    
    total_time = time.time() - start_time
    
    # Save results
    excel_file = save_to_excel_strict(excel_data, summary, file_name)
    json_file = save_to_json_strict(excel_data, summary, contract_fields, service_rows, file_name)
    
    # Display summary
    print("\n" + "=" * 60)
    print("🎉 EXTRACTION COMPLETE (STRICT MODE)")
    print("=" * 60)
    
    print(f"⏱️ Total Time: {total_time:.1f} seconds")
    print(f"📊 Success Rate: {summary['success_rate']}")
    print(f"📋 Fields Extracted: {summary['extracted_fields']}/{summary['total_fields']}")
    print(f"🏥 Service Rows Created: {summary['service_rows']}")
    
    if excel_file:
        print(f"💾 Excel file: {excel_file}")
    if json_file:
        print(f"💾 JSON file: {json_file}")
    
    # Preview data (first 3 rows)
    if excel_data:
        print(f"\n📋 Data Preview (first 3 rows, key columns):")
        print("-" * 100)
        
        preview_data = excel_data[:3]
        for i, row in enumerate(preview_data):
            print(f"\nRow {i+1}:")
            key_fields = ["PROVIDER NAME", "CIS CTRCT ID", "LOB IND", "SERVICE TYPE", "PRICE RATE"]
            for field in key_fields:
                value = row.get(field, "N/A")
                print(f"  {field}: {value}")
    
    return {
        "summary": summary,
        "excel_data": excel_data,
        "contract_fields": contract_fields,
        "service_rows": service_rows,
        "excel_file": excel_file,
        "json_file": json_file
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

