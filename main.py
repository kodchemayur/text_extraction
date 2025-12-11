import os
import faiss
import numpy as np
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import ollama
import pandas as pd 
import shutil 
import json
import re
import time
from pathlib import Path
from datetime import datetime
import pickle

# --- GLOBAL SETUP & CONSTANTS ---
OLLAMA_CLIENT = ollama.Client() 

# --- Model Selection ---
TEXT_LLM_MODEL = "llama3.1:8b"  # or "llama3.2:3b" for faster processing
EMBEDDING_MODEL = "nomic-embed-text"

# --- CCG2.0 Fields - EXACT ORDER FOR EXCEL ---
ALL_FIELD_NAMES = [
    # === CONTRACT & PROVIDER IDENTIFICATION ===
    "CIS CTRCT ID", "ATTACHMENT ID", "FILE NAME", "FILE EXTENSION", "CIS TYPE",
    "CIS TYPE DESCRIPTION", "PROVIDER NAME", "NPI", "TAX ID", "PROVZIPCODE",
    "TAXONOMYCODE", "EFFECTIVE FROM DATE", "EFFECTIVE TO DATE", "PLACEOFSERV",
    "PROVIDER SPECIALITY",
    
    # === SERVICE LEVEL FIELDS ===
    "LOB IND", "SERVICE TYPE", "SERVICE DESC", "SERVICES", "AGE GROUP", "CODES",
    "GROUPER", "PRICE RATE", "REIMBURSEMENT AMT", "REIMBURSEMENT RATE", 
    "REIMBURSEMENT METHODOLOGY", "HEALTH BENEFIT PLANS",
    
    # === INDICATOR FIELDS ===
    "REVENUE CD IND", "DRG CD IND", "CPT IND", "HCPCS IND", "ICD CD IND",
    "DIAGNOSIS CD IND", "MODIFIER CD IND", "GROUPER IND", "APC IND", 
    "EXCLUSION IND", "MSR IND", "BILETRAL PROCEDURE IND", 
    "EXCLUDE FROM TRANSFER IND", "EXCLUDE FROM STOPLOSS IND",
    
    # === DISCHARGE & PAYMENT FIELDS ===
    "DISCHARGESTATUSCODE", "ALOSGLOS", "TRANSFER RATE", "APPLIEDTRANSFERCASE",
    "ISTHRESHOLD", "ISCAPAMOUNT", "MULTIPLERMETHODS", "METHOD OF PAYMENT",
    
    # === ADDITIONAL FIELDS ===
    "ADDITIONAL NOTES", "OTHER FLAT FEE", "SURG FLAT FEE", 
    "AND OR OPERATOR", "OPERATOR CODE TYPE",
    
    # === NLP PROCESSING FIELDS ===
    "NLP USER ID", "NLP PROCESS TIMESTAMP", "NLP ERROR COMMENTS"
]

# Group fields for reference (not used in output order)
CONTRACT_LEVEL_FIELDS = [
    "CIS CTRCT ID", "ATTACHMENT ID", "FILE NAME", "FILE EXTENSION", "CIS TYPE",
    "CIS TYPE DESCRIPTION", "PROVIDER NAME", "NPI", "TAX ID", "PROVZIPCODE",
    "TAXONOMYCODE", "EFFECTIVE FROM DATE", "EFFECTIVE TO DATE", "PLACEOFSERV",
    "PROVIDER SPECIALITY", "NLP USER ID", "NLP PROCESS TIMESTAMP", "NLP ERROR COMMENTS"
]

SERVICE_LEVEL_FIELDS = [
    "LOB IND", "SERVICE TYPE", "SERVICE DESC", "SERVICES", "AGE GROUP", "CODES",
    "GROUPER", "PRICE RATE", "REIMBURSEMENT AMT", "REIMBURSEMENT RATE", 
    "REIMBURSEMENT METHODOLOGY", "HEALTH BENEFIT PLANS"
]

INDICATOR_FIELDS = [
    "REVENUE CD IND", "DRG CD IND", "CPT IND", "HCPCS IND", "ICD CD IND",
    "DIAGNOSIS CD IND", "MODIFIER CD IND", "GROUPER IND", "APC IND", 
    "EXCLUSION IND", "MSR IND", "BILETRAL PROCEDURE IND", 
    "EXCLUDE FROM TRANSFER IND", "EXCLUDE FROM STOPLOSS IND"
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_string_convert(value):
    """Safely convert any value to string - NO ASSUMPTIONS"""
    if value is None:
        return "N/A"
    elif isinstance(value, (list, dict)):
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return json.dumps(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else "N/A"
    else:
        str_val = str(value).strip()
        return str_val if str_val else "N/A"

def call_ollama_chat(system_prompt, user_prompt, model=TEXT_LLM_MODEL, format="json", temperature=0):
    """Call Ollama with SYSTEM + USER prompts"""
    try:
        print(f"🤖 Calling Ollama model: {model}")
        
        response = OLLAMA_CLIENT.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            format=format,
            options={"temperature": temperature, "num_predict": 4000}
        )
        
        content = response['message']['content'].strip()
        
        # Save raw response
        debug_dir = "./debug"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%H%M%S")
        with open(os.path.join(debug_dir, f"ollama_response_{timestamp}.txt"), 'w', encoding='utf-8') as f:
            f.write(f"=== SYSTEM PROMPT ===\n{system_prompt[:1000]}...\n\n")
            f.write(f"=== USER PROMPT ===\n{user_prompt[:1000]}...\n\n")
            f.write(f"=== RESPONSE ===\n{content}")
        
        print(f"📄 Ollama response: {len(content)} chars")
        
        # Clean markdown
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        return content.strip()
    except Exception as e:
        print(f"❌ Ollama API error: {e}")
        raise

def call_ollama_embeddings(texts, model=EMBEDDING_MODEL):
    """Generate embeddings using Ollama"""
    embeddings = []
    for text in texts:
        try:
            response = OLLAMA_CLIENT.embeddings(model=model, prompt=text)
            embeddings.append(response['embedding'])
        except Exception:
            # Return zero vector as fallback
            embeddings.append([0.0] * 768)
    return embeddings

class OllamaEmbeddingFunction:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model_name = model_name
        self.dimension = 768

    def __call__(self, texts):
        return call_ollama_embeddings(texts, self.model_name)

# ============================================================================
# VECTOR STORE FUNCTIONS
# ============================================================================

def initialize_faiss_store(dimension=768, index_path="./faiss_index"):
    """Initialize FAISS vector store"""
    os.makedirs(index_path, exist_ok=True)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # Store metadata and documents
    metadata = []
    documents = []
    ids = []
    
    # Load existing index if available
    index, metadata, documents, ids = load_faiss_index(index_path, dimension)
    
    return {
        "index": index,
        "metadata": metadata,
        "documents": documents,
        "ids": ids,
        "index_path": index_path,
        "dimension": dimension
    }

def load_faiss_index(index_path, dimension):
    """Load FAISS index from disk"""
    index_file = os.path.join(index_path, "index.faiss")
    metadata_file = os.path.join(index_path, "metadata.pkl")
    documents_file = os.path.join(index_path, "documents.pkl")
    ids_file = os.path.join(index_path, "ids.pkl")
    
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        
        metadata = []
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
        
        documents = []
        if os.path.exists(documents_file):
            with open(documents_file, 'rb') as f:
                documents = pickle.load(f)
        
        ids = []
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                ids = pickle.load(f)
        
        return index, metadata, documents, ids
    
    # Return new index if not found
    return faiss.IndexFlatL2(dimension), [], [], []

def save_faiss_index(vector_store):
    """Save FAISS index to disk"""
    try:
        index_path = vector_store["index_path"]
        
        # Save FAISS index
        faiss.write_index(vector_store["index"], os.path.join(index_path, "index.faiss"))
        
        # Save metadata
        with open(os.path.join(index_path, "metadata.pkl"), 'wb') as f:
            pickle.dump(vector_store["metadata"], f)
        
        # Save documents
        with open(os.path.join(index_path, "documents.pkl"), 'wb') as f:
            pickle.dump(vector_store["documents"], f)
        
        # Save IDs
        with open(os.path.join(index_path, "ids.pkl"), 'wb') as f:
            pickle.dump(vector_store["ids"], f)
        
        return True
    except Exception as e:
        print(f"Error saving FAISS index: {e}")
        return False

def add_documents_to_faiss(vector_store, documents, metadatas=None, ids=None):
    """Add documents to vector store"""
    if not documents:
        return False
    
    # Generate embeddings
    embedding_fn = OllamaEmbeddingFunction()
    embeddings = embedding_fn(documents)
    
    # Convert to numpy array
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Add to FAISS index
    vector_store["index"].add(embeddings_np)
    
    # Store metadata and documents
    for i, doc in enumerate(documents):
        vector_store["documents"].append(doc)
        
        if metadatas and i < len(metadatas):
            vector_store["metadata"].append(metadatas[i])
        else:
            vector_store["metadata"].append({"id": f"doc_{len(vector_store['documents'])}"})
        
        if ids and i < len(ids):
            vector_store["ids"].append(ids[i])
        else:
            vector_store["ids"].append(f"doc_{len(vector_store['documents'])}")
    
    # Save index
    save_faiss_index(vector_store)
    return True

def query_faiss(vector_store, query_text, n_results=10):
    """Query the vector database"""
    try:
        # Generate embedding for query
        embedding_fn = OllamaEmbeddingFunction()
        query_embedding = embedding_fn([query_text])[0]
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS
        distances, indices = vector_store["index"].search(
            query_embedding_np, 
            min(n_results, len(vector_store["documents"]))
        )
        
        # Prepare results
        results = {
            'documents': [],
            'metadatas': [],
            'distances': distances[0].tolist(),
            'indices': indices[0].tolist()
        }
        
        for idx in indices[0]:
            if idx >= 0 and idx < len(vector_store["documents"]):
                results['documents'].append(vector_store["documents"][idx])
                if idx < len(vector_store["metadata"]):
                    results['metadatas'].append(vector_store["metadata"][idx])
        
        return results
    except Exception as e:
        print(f"Error querying FAISS index: {e}")
        return {'documents': [], 'metadatas': [], 'distances': [], 'indices': []}

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def process_document(file_path, image_dir="extracted_images"):
    """Process PDF and extract text/images"""
    print("\n📄 Stage 1: Processing Document...")
    
    try:
        # Cleanup and create directories
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
        os.makedirs(image_dir, exist_ok=True)
        
        # Extract elements from PDF
        extracted_elements = partition_pdf(
            filename=file_path,
            strategy="ocr_only",
            ocr_languages=["eng"],
            ocr_agent = "paddle_ocr",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            pdf_extract_images_path=image_dir,
            extract_tables=True,
            chunking_strategy="by_title"
        )
        
        # Filter text elements
        text_elements = [e for e in extracted_elements 
                       if hasattr(e, 'text') and e.text and e.text.strip()]
        
        # Chunk by title
        chunks = chunk_by_title(
            elements=text_elements,
            max_characters=800,  # Smaller chunks for better retrieval
            new_after_n_chars=600,
            combine_text_under_n_chars=150,
            overlap=100  # Add overlap
        )
        
        # Store chunks in raw_data_store
        raw_data_store = {}
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.text
            page_num = getattr(chunk.metadata, 'page_number', 1)
            raw_id = f"chunk_{idx}"
            raw_data_store[raw_id] = {
                "content": chunk_text,
                "page": page_num,
                "type": "text"
            }
        
        # Store image references
        try:
            for i, img_path in enumerate(sorted(Path(image_dir).glob("*"))):
                raw_data_store[f"image_{i}"] = {
                    "content": f"[IMAGE_FILE] {img_path.name}",
                    "page": None,
                    "type": "image",
                    "path": str(img_path)
                }
        except Exception:
            pass
        
        print(f"✅ Created {len(chunks)} text chunks")
        return {
            "success": True,
            "chunks": chunks,
            "raw_data_store": raw_data_store,
            "image_dir": image_dir
        }
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks": [],
            "raw_data_store": {},
            "image_dir": image_dir
        }

def store_in_vector_db(chunks, db_path="./faiss_contract_db"):
    """Store chunks in vector database"""
    print("\n💾 Stage 2: Storing in Vector Database...")
    
    if not chunks:
        print("❌ No chunks to store")
        return {"success": False, "vector_store": None}
    
    # Initialize vector store
    vector_store = initialize_faiss_store(dimension=768, index_path=db_path)
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.text
        page_num = getattr(chunk.metadata, 'page_number', 1)
        
        documents.append(chunk_text)
        metadatas.append({
            "page_number": page_num,
            "chunk_id": f"chunk_{idx}",
            "type": "text"
        })
        ids.append(f"doc_{idx}")
    
    # Add to FAISS vector store
    success = add_documents_to_faiss(
        vector_store=vector_store,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    if success:
        print(f"✅ Stored {len(documents)} chunks in FAISS vector database")
        return {"success": True, "vector_store": vector_store}
    else:
        print("❌ Failed to store chunks in FAISS")
        return {"success": False, "vector_store": None}

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

if __name__ == "__main__":
    # Get file path
    file_path = "sample-humana1.pdf"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Please provide a valid PDF file path.")
        print("\nUsage:")
        print(f"  python {__file__} <path_to_pdf_file>")
        exit(1)
    
    # Run extraction
    results = run_extraction_pipeline_strict(file_path)
    
    if results:
        print("\n✅ Extraction completed successfully!")
        print(f"📊 Final extraction rate: {results['summary']['success_rate']}")
    else:
        print("\n❌ Extraction failed!")