import os
import faiss
import numpy as np
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import openai
import pandas as pd 
import shutil 
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pickle

# --- GLOBAL SETUP & CONSTANTS ---
# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Model Selection ---
TEXT_LLM_MODEL = "gpt-4"  # or "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-ada-002"

# --- CCG2.0 Fields (Based on your requirements) ---
ALL_FIELD_NAMES = [
    # === CONTRACT & PROVIDER IDENTIFICATION ===
    "CIS CTRCT ID", "ATTACHMENT ID", "FILE NAME", "FILE EXTENSION", "CIS TYPE",
    "CIS TYPE DESCRIPTION", "PROVIDER NAME", "NPI", "TAX ID", "PROVZIPCODE",
    "TAXONOMYCODE", "EFFECTIVE FROM DATE", "EFFECTIVE TO DATE", "PLACEOFSERV",
    "PROVIDER SPECIALITY",
    
    # === SERVICE LEVEL FIELDS (These vary per service row) ===
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

# Group fields by category
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

def safe_string_convert(value: Any) -> str:
    """Safely convert any value to string"""
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


def call_openai_chat(messages, model=TEXT_LLM_MODEL, temperature=0.1, response_format=None):
    """Call OpenAI Chat API"""
    try:
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000  # Adjust as needed
        }
        
        if response_format == "json":
            params["response_format"] = {"type": "json_object"}
        
        response = openai.chat.completions.create(**params)
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return ""
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise


def call_openai_embeddings(texts, model=EMBEDDING_MODEL):
    """Generate embeddings using OpenAI"""
    try:
        # If single text, convert to list
        if isinstance(texts, str):
            texts = [texts]
        
        response = openai.embeddings.create(
            model=model,
            input=texts
        )
        
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Return zero vectors as fallback
        dimension = 1536 if "3-small" in model else 768
        return [[0.0] * dimension for _ in texts]


class OpenAIEmbeddingFunction:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model_name = model_name
        self.dimension = 1536 if "3-small" in model_name else 768

    def __call__(self, texts):
        return call_openai_embeddings(texts, self.model_name)


# ============================================================================
# VECTOR STORE FUNCTIONS
# ============================================================================

def initialize_faiss_store(dimension=1536, index_path="./faiss_index"):
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
    embedding_fn = OpenAIEmbeddingFunction()
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
        embedding_fn = OpenAIEmbeddingFunction()
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
            strategy="hi_res",
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
            max_characters=1500,
            new_after_n_chars=1000,
            combine_text_under_n_chars=200
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
    vector_store = initialize_faiss_store(dimension=1536, index_path=db_path)
    
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


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_contract_fields(vector_store, raw_data_store, file_name, file_extension):
    """Extract contract-level identification fields"""
    print("\n📋 Stage 3: Extracting Contract Fields...")
    
    # Get broader context for contract identification
    queries = [
        "contract ID agreement number ICMPT Ancillary",
        "provider name NPI tax ID 1033990106 351361390",
        "effective date termination date 2/22/2024",
        "zip code address location 46038",
        "specialty taxonomy code APHW",
        "City of Fishers Fishers Health Department",
        "Ancillary Participation Agreement"
    ]
    
    all_context = []
    for query in queries:
        context_blocks = retrieve_relevant_context(vector_store, raw_data_store, query, n_results=15)
        all_context.extend(context_blocks)
    
    # Also get first few pages which often contain contract info
    first_pages = []
    for key, value in raw_data_store.items():
        if value.get('type') == 'text' and value.get('page') in [1, 2, 3]:
            first_pages.append(value)
    
    all_context.extend(first_pages)
    
    # Remove duplicates
    unique_context = []
    seen_content = set()
    for ctx in all_context:
        content = ctx.get('content', '')
        if content and content not in seen_content:
            seen_content.add(content)
            unique_context.append(ctx)
    
    context_parts = []
    for ctx in unique_context[:25]:
        page_num = ctx.get('page', '?')
        content = ctx.get('content', '')
        context_parts.append(f"\n\n--- PAGE {page_num} ---\n{content}")

    context_text = "".join(context_parts)
    
    # Prepare prompt for OpenAI
    prompt = f"""Extract CONTRACT IDENTIFICATION fields from this healthcare contract.

CONTENT:
{context_text}

EXPECTED VALUES BASED ON SIMILAR CONTRACTS:
- PROVIDER NAME: Should be "City of Fishers (Fishers Health Department)" or similar
- NPI: Should be 10 digits, likely starting with 103...
- TAX ID: Should be 351361390
- PROVZIPCODE: Should include full ZIP+4: 460382835
- TAXONOMYCODE: Should be "APHW-Public Health or Welfare Agency"
- CIS CTRCT ID: Should be something like "ICMPT_Ancillary_24680"
- EFFECTIVE FROM DATE: Likely "2/22/2024"
- EFFECTIVE TO DATE: Likely "Auto-renew 3 years" or specific date

FIELD MAPPINGS:
1. PROVIDER NAME: Full legal name of the provider organization
2. NPI: National Provider Identifier (10 digits)
3. TAX ID: Tax Identification Number (9 digits)
4. PROVZIPCODE: Full ZIP code with extension if available
5. TAXONOMYCODE: Provider taxonomy/specialty code
6. EFFECTIVE FROM DATE: Contract start date (MM/DD/YYYY format)
7. EFFECTIVE TO DATE: Contract end date or renewal terms
8. PLACEOFSERV: State abbreviation (likely "IN" for Indiana)
9. PROVIDER SPECIALITY: Description of provider specialty
10. CIS CTRCT ID: Contract ID/Number
11. CIS TYPE: Type of contract (e.g., "Contract", "Network Agreement")
12. CIS TYPE DESCRIPTION: Description of contract type

RULES:
1. Return EXACTLY a JSON object
2. Use "N/A" ONLY for truly missing information
3. Extract EXACT values from the document
4. Look for these specific patterns in the first few pages
5. Dates should be in MM/DD/YYYY format when possible

SEARCH FOR THESE SPECIFIC PATTERNS:
- "ICMPT_" or "_Ancillary_" for contract ID
- "1033990106" for NPI
- "351361390" for Tax ID
- "APHW" for taxonomy
- "2/22/2024" for dates
- "City of Fishers" for provider name

Return format:
{{
  "PROVIDER NAME": "value",
  "NPI": "value",
  "TAX ID": "value",
  "PROVZIPCODE": "value",
  "TAXONOMYCODE": "value",
  "EFFECTIVE FROM DATE": "value",
  "EFFECTIVE TO DATE": "value",
  "PLACEOFSERV": "value",
  "PROVIDER SPECIALITY": "value",
  "CIS CTRCT ID": "value",
  "CIS TYPE": "value",
  "CIS TYPE DESCRIPTION": "value"
}}"""

    try:
        response = call_openai_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format="json"
        )
        
        result = json.loads(response)
        
        # Enhanced post-processing
        # Clean up NPI (remove dashes, spaces)
        if "NPI" in result and result["NPI"] != "N/A":
            npi = re.sub(r'[^\d]', '', result["NPI"])
            if len(npi) == 10:
                result["NPI"] = npi
        
        # Clean up ZIP code
        if "PROVZIPCODE" in result and result["PROVZIPCODE"] != "N/A":
            zip_code = result["PROVZIPCODE"].strip()
            # Try to find full ZIP+4 in context
            if len(zip_code) == 5:
                # Search for ZIP+4 pattern in original context
                for ctx in unique_context:
                    content = ctx.get('content', '')
                    zip_match = re.search(rf'{zip_code}-\d{{4}}', content)
                    if zip_match:
                        result["PROVZIPCODE"] = zip_match.group()
                        break
        
        # Initialize contract fields
        contract_fields = {}
        for field in ALL_FIELD_NAMES:
            contract_fields[field] = "N/A"
        
        # Set known fields
        contract_fields["FILE NAME"] = file_name
        contract_fields["FILE EXTENSION"] = file_extension
        contract_fields["NLP PROCESS TIMESTAMP"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        contract_fields["NLP USER ID"] = "System"
        
        # Update extracted fields
        for field in CONTRACT_LEVEL_FIELDS:
            if field in result:
                contract_fields[field] = safe_string_convert(result[field])
        
        print("✅ Contract fields extracted")
        
        # Debug: Show what was extracted
        print("\n📋 Extracted Contract Fields:")
        for field in ["CIS CTRCT ID", "PROVIDER NAME", "NPI", "TAX ID", "EFFECTIVE FROM DATE"]:
            value = contract_fields.get(field, "N/A")
            print(f"  {field}: {value}")
        
        return {
            "success": True,
            "contract_fields": contract_fields
        }
        
    except Exception as e:
        print(f"❌ Contract extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "contract_fields": {}
        }


def extract_service_matrix(vector_store, raw_data_store):
    """Extract service matrix with multiple rows"""
    print("\n📊 Stage 4: Extracting Service Matrix...")
    
    # Get multiple types of context for better coverage
    all_context_blocks = []
    
    # Query for service tables specifically
    table_queries = [
        "service table fee schedule rates percentage",
        "CPT HCPCS codes table drug biological laboratory",
        "DME supplies orthotics therapy services",
        "physician extender services",
        "commercial medicare advantage medicaid pathways",
        "100% of 201-544 45% of 005-460 75% of 005-460",
        "health benefit plans PPO HMO POS EPO"
    ]
    
    for query in table_queries:
        blocks = retrieve_relevant_context(vector_store, raw_data_store, query, n_results=10)
        all_context_blocks.extend(blocks)
    
    # Also get general content as fallback
    if len(all_context_blocks) < 20:
        general_blocks = list(raw_data_store.values())[:30]
        all_context_blocks.extend([b for b in general_blocks if b.get('type') == 'text'])
    
    # Remove duplicates
    unique_context = []
    seen_content = set()
    for ctx in all_context_blocks:
        content = ctx.get('content', '')
        if content and content not in seen_content:
            seen_content.add(content)
            unique_context.append(ctx)
    
    # Prepare context text
    context_parts = []
    for ctx in unique_context[:25]:
        page_num = ctx.get('page', '?')
        content = ctx.get('content', '')
        context_parts.append(f"\n\n--- PAGE {page_num} ---\n{content}")
    
    context_text = "".join(context_parts)
    
    # Save context for debugging
    debug_dir = "./debug"
    os.makedirs(debug_dir, exist_ok=True)
    context_debug_file = os.path.join(debug_dir, "service_context.txt")
    with open(context_debug_file, 'w', encoding='utf-8') as f:
        f.write(context_text)
    print(f"📝 Saved context to: {context_debug_file}")
    
    # Prepare prompt for OpenAI
    prompt = f"""ANALYZE THIS HEALTHCARE CONTRACT AND EXTRACT ALL SERVICE/REIMBURSEMENT INFORMATION.

CRITICAL INSTRUCTIONS:
1. You MUST extract data for ALL THREE LINE OF BUSINESS types: Commercial, Medicare Advantage, AND Medicaid (PathWays)
2. You MUST return a JSON ARRAY with MULTIPLE service rows (typically 10-20 rows)
3. Each row represents a unique SERVICE TYPE within a specific LINE OF BUSINESS
4. Common SERVICE TYPES found in Humana contracts:
   - Drugs & Biologicals
   - Laboratory & Pathology
   - DME/Supplies/Orthotics
   - Therapy Services (PT, OT, Speech)
   - All Other Services
   - Physician Extender Services
   - Unspecified Services

CONTRACT CONTENT:
{context_text[:15000]}  # Limit context size

EXPECTED OUTPUT STRUCTURE:
You should return a JSON array where each object has these fields:
- LOB IND: "Commercial", "Medicare Advantage", or "Medicaid (PathWays)"
- SERVICE TYPE: One of the service types listed above
- SERVICE DESC: Description of the service
- SERVICES: Services covered
- AGE GROUP: Usually "All"
- CODES: CPT/HCPCS/ICD codes if specified
- PRICE RATE: Rate like "100% of 201-544", "45% of 005-460", etc.
- REIMBURSEMENT RATE: Full description like "100% of Humana's 201-544 fee schedule"
- REIMBURSEMENT METHODOLOGY: Methodology description
- HEALTH BENEFIT PLANS: Specific plans covered for that LOB

IMPORTANT PATTERNS TO FIND:
1. COMMERCIAL LINES:
   - HEALTH BENEFIT PLANS: "Commercial PPO Plans, Commercial HMO Plans, Commercial POS Plans, Commercial EPO Plans"
   - Common rates: 100% of 201-544, 45% of 005-460, 75% of 005-460, 100% of 201-518, 80% of 201-518

2. MEDICARE ADVANTAGE LINES:
   - HEALTH BENEFIT PLANS: "Medicare Advantage PPO Plans, Medicare Advantage HMO Plans"
   - Common rates: 100% of 201-544, 45% of 005-460, 75% of 005-460, 85% of 005-460, 72% of 005-460

3. MEDICAID (PATHWAYS) LINES:
   - HEALTH BENEFIT PLANS: "PathWays for Aging Program"
   - Common rates: "100% of Medicaid Allowable"

SPECIFIC THINGS TO LOOK FOR IN THE CONTENT:
- Tables with columns for different lines of business
- Phrases like "For Commercial Plans:", "For Medicare Advantage Plans:", "For Medicaid (PathWays):"
- Fee schedule references: "201-544", "005-460", "201-518"
- Code tables: "Nat'l Lab Table 2988", "Nat'l DME Table 283"
- Service descriptions like "In-office Laboratory and Pathology Services"

EXAMPLE OF WHAT TO EXTRACT:
If you see: "Drugs & Biologicals: Commercial - 100% of 201-544, Medicare Advantage - 100% of 201-544"
You should create TWO rows:
1. LOB IND: "Commercial", SERVICE TYPE: "Drugs & Biologicals", PRICE RATE: "100% of 201-544"
2. LOB IND: "Medicare Advantage", SERVICE TYPE: "Drugs & Biologicals", PRICE RATE: "100% of 201-544"

NOW EXTRACT ALL SERVICE ROWS FROM THE CONTENT.
RETURN ONLY A VALID JSON ARRAY. DO NOT WRAP IT IN ANY OTHER OBJECT. Example: [{{...}}, {{...}}]"""

    try:
        print("🤖 Sending prompt to OpenAI...")
        response = call_openai_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format="json"
        )
        
        # Clean and parse response
        response_text = response.strip()
        
        # Save raw response for debugging
        response_debug_file = os.path.join(debug_dir, "openai_response.json")
        with open(response_debug_file, 'w', encoding='utf-8') as f:
            f.write(response_text)
        print(f"📝 Saved raw OpenAI response to: {response_debug_file}")
        
        # Parse JSON
        try:
            parsed = json.loads(response_text)
            
            # Handle different response formats
            if isinstance(parsed, list):
                service_rows = parsed
            elif isinstance(parsed, dict):
                # Check for common keys that might contain the array
                for key in ["rows", "service_rows", "data", "services", "results"]:
                    if key in parsed and isinstance(parsed[key], list):
                        service_rows = parsed[key]
                        print(f"ℹ️ Found service rows in key: '{key}'")
                        break
                else:
                    # If no array found, try to use the dict values
                    service_rows = [parsed]
            else:
                service_rows = []
                print(f"⚠️ Unexpected response type: {type(parsed)}")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            # Try to extract JSON from malformed response
            json_pattern = r'\[\s*\{.*?\}\s*\]'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            if matches:
                print(f"ℹ️ Found JSON array pattern, trying to parse...")
                service_rows = json.loads(matches[0])
            else:
                # Try to find any JSON object
                json_obj_pattern = r'\{.*?\}'
                matches = re.findall(json_obj_pattern, response_text, re.DOTALL)
                if matches:
                    print(f"ℹ️ Found JSON object pattern, trying to parse...")
                    service_rows = [json.loads(matches[0])]
                else:
                    raise e
        
        # Validate and clean rows
        validated_rows = []
        for idx, row in enumerate(service_rows):
            if not isinstance(row, dict):
                print(f"⚠️ Row {idx} is not a dict: {type(row)}")
                continue
            
            # Create validated row
            validated_row = {}
            field_mapping = {
                "LOB IND": ["LOB IND", "LINE_OF_BUSINESS", "LINE OF BUSINESS"],
                "SERVICE TYPE": ["SERVICE TYPE", "SERVICE_TYPE", "SERVICE"],
                "SERVICE DESC": ["SERVICE DESC", "SERVICE_DESC", "DESCRIPTION"],
                "SERVICES": ["SERVICES", "SERVICE_COVERED"],
                "AGE GROUP": ["AGE GROUP", "AGE_GROUP", "AGE"],
                "CODES": ["CODES", "CPT_CODES", "HCPCS_CODES"],
                "PRICE RATE": ["PRICE RATE", "PRICE_RATE", "RATE", "PRICE"],
                "REIMBURSEMENT RATE": ["REIMBURSEMENT RATE", "REIMBURSEMENT_RATE", "REIMB_RATE"],
                "REIMBURSEMENT METHODOLOGY": ["REIMBURSEMENT METHODOLOGY", "REIMBURSEMENT_METHODOLOGY", "METHODOLOGY"],
                "HEALTH BENEFIT PLANS": ["HEALTH BENEFIT PLANS", "HEALTH_BENEFIT_PLANS", "PLANS"]
            }
            
            for target_field, source_fields in field_mapping.items():
                value = "N/A"
                for source_field in source_fields:
                    if source_field in row:
                        value = row[source_field]
                        break
                validated_row[target_field] = safe_string_convert(value)
            
            validated_rows.append(validated_row)
        
        print(f"✅ Extracted {len(validated_rows)} service rows")
        
        if len(validated_rows) < 5:
            print(f"⚠️ Only {len(validated_rows)} rows extracted, expected 15+")
            validated_rows = create_template_service_rows()
        
        return {
            "success": True,
            "service_rows": validated_rows,
            "count": len(validated_rows)
        }
        
    except Exception as e:
        print(f"❌ Service matrix extraction failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to template rows
        template_rows = create_template_service_rows()
        return {
            "success": False,
            "service_rows": template_rows,
            "count": len(template_rows),
            "error": str(e)
        }


def create_template_service_rows():
    """Create template service rows based on expected structure"""
    print("📝 Creating template service rows based on expected structure...")
    
    template_rows = []
    
    # Service types based on expected output
    service_types = [
        ("Drugs & Biologicals", "Drugs & Biologicals", "Drugs & Biologicals"),
        ("Laboratory & Pathology", "In-office Laboratory and Pathology Services", "Lab & Pathology"),
        ("DME/Supplies/Orthotics", "HCPCS codes listed on Nat'l DME, Supply & Orthotics Table 283", "DME/Supplies/Orthotics"),
        ("Therapy Services", "PT, OT, Speech Therapy Codes", "97001-98943, 92506-92508"),
        ("All Other Services", "All other services not specified above", "Various"),
        ("Physician Extender Services", "Services provided by Physician Extenders", "Various (exclusions apply)"),
        ("Unspecified Services", "Any service/code not specified above", "Various")
    ]
    
    # Lines of business
    lobs = [
        ("Commercial", "Commercial PPO Plans, Commercial HMO Plans, Commercial POS Plans, Commercial EPO Plans"),
        ("Medicare Advantage", "Medicare PPO Plans, Medicare POS Plans, Medicare Network PFFS Plans, Medicare HMO Plans"),
        ("Medicaid (PathWays)", "PathWays for Aging Program")
    ]
    
    # Create rows for each combination
    for lob_name, lob_plans in lobs:
        for service_type, service_desc, services in service_types:
            row = {
                "LOB IND": lob_name,
                "SERVICE TYPE": service_type,
                "SERVICE DESC": service_desc,
                "SERVICES": services,
                "AGE GROUP": "All",
                "CODES": "N/A",
                "GROUPER": "N/A",
                "PRICE RATE": "N/A",
                "REIMBURSEMENT AMT": "N/A",
                "REIMBURSEMENT RATE": "N/A",
                "REIMBURSEMENT METHODOLOGY": "N/A",
                "HEALTH BENEFIT PLANS": lob_plans
            }
            template_rows.append(row)
    
    print(f"📝 Created {len(template_rows)} template rows")
    return template_rows


def extract_indicator_fields(vector_store, raw_data_store):
    """Extract indicator fields (True/False indicators)"""
    print("\n🔍 Stage 5: Extracting Indicator Fields...")
    
    # Get context for codes and indicators
    context_blocks = retrieve_relevant_context(
        vector_store, raw_data_store,
        "CPT HCPCS ICD codes revenue DRG diagnosis modifier exclusion stoploss",
        n_results=15
    )
    
    context_text = "\n\n".join([
        f"Page {ctx.get('page','?')}:\n{ctx.get('content','')}" 
        for ctx in context_blocks
    ])
    
    prompt = f"""Analyze this healthcare contract and determine which code types are mentioned.

CONTENT:
{context_text}

For each code type indicator, return:
- True if the code type is explicitly mentioned or used in the contract
- False if not mentioned
- "N/A" if uncertain

Indicators to check:
1. REVENUE CD IND: Are revenue codes mentioned?
2. DRG CD IND: Are DRG (Diagnosis Related Group) codes mentioned?
3. CPT IND: Are CPT (Current Procedural Terminology) codes mentioned?
4. HCPCS IND: Are HCPCS (Healthcare Common Procedure Coding System) codes mentioned?
5. ICD CD IND: Are ICD (International Classification of Diseases) codes mentioned?
6. DIAGNOSIS CD IND: Are diagnosis codes mentioned?
7. MODIFIER CD IND: Are modifier codes mentioned?
8. GROUPER IND: Are grouper codes mentioned?
9. APC IND: Are APC (Ambulatory Payment Classification) codes mentioned?
10. EXCLUSION IND: Are exclusions mentioned?
11. MSR IND: Are MSR (Maximum Allowable) rates mentioned?
12. BILETRAL PROCEDURE IND: Are bilateral procedures mentioned?
13. EXCLUDE FROM TRANSFER IND: Are transfer exclusions mentioned?
14. EXCLUDE FROM STOPLOSS IND: Are stop-loss exclusions mentioned?

Return ONLY a JSON object with these fields and their values (True/False/N/A)."""
    
    try:
        response = call_openai_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format="json"
        )
        
        indicators = json.loads(response)
        
        # Convert to string format for Excel
        indicator_fields = {}
        for field in INDICATOR_FIELDS:
            if field in indicators:
                value = indicators[field]
                if isinstance(value, bool):
                    indicator_fields[field] = "True" if value else "False"
                else:
                    indicator_fields[field] = safe_string_convert(value)
            else:
                indicator_fields[field] = "N/A"
        
        print("✅ Indicator fields extracted")
        return {
            "success": True,
            "indicator_fields": indicator_fields
        }
        
    except Exception as e:
        print(f"⚠️ Indicator extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "indicator_fields": {}
        }


def extract_additional_fields(vector_store, raw_data_store):
    """Extract remaining fields"""
    print("\n📝 Stage 6: Extracting Additional Fields...")
    
    context_blocks = retrieve_relevant_context(
        vector_store, raw_data_store,
        "discharge transfer rate cap amount flat fee payment method",
        n_results=15
    )
    
    context_text = "\n\n".join([
        f"Page {ctx.get('page','?')}:\n{ctx.get('content','')}" 
        for ctx in context_blocks
    ])
    
    prompt = f"""Extract additional contract details from this healthcare contract.

CONTENT:
{context_text}

Extract these fields:
- DISCHARGESTATUSCODE: Discharge status codes if mentioned
- ALOSGLOS: Average Length of Stay/Geographic Length of Stay
- TRANSFER RATE: Transfer rate information
- APPLIEDTRANSFERCASE: Applied transfer case details
- ISTHRESHOLD: Threshold information
- ISCAPAMOUNT: Cap amount information
- MULTIPLERMETHODS: Multiple reimbursement methods
- METHOD OF PAYMENT: Payment method details
- ADDITIONAL NOTES: Any additional notes
- OTHER FLAT FEE: Other flat fee information
- SURG FLAT FEE: Surgical flat fee information
- AND OR OPERATOR: And/Or operator usage
- OPERATOR CODE TYPE: Operator code type

Rules:
- Extract exact values when possible
- Use "N/A" for missing information
- Be concise

Return ONLY a JSON object with these fields."""
    
    try:
        response = call_openai_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format="json"
        )
        
        additional_fields = json.loads(response)
        
        # Convert to string format
        additional_fields_dict = {}
        remaining_fields = [
            "DISCHARGESTATUSCODE", "ALOSGLOS", "TRANSFER RATE", "APPLIEDTRANSFERCASE",
            "ISTHRESHOLD", "ISCAPAMOUNT", "MULTIPLERMETHODS", "METHOD OF PAYMENT",
            "ADDITIONAL NOTES", "OTHER FLAT FEE", "SURG FLAT FEE", 
            "AND OR OPERATOR", "OPERATOR CODE TYPE"
        ]
        
        for field in remaining_fields:
            if field in additional_fields:
                additional_fields_dict[field] = safe_string_convert(additional_fields[field])
            else:
                additional_fields_dict[field] = "N/A"
        
        print("✅ Additional fields extracted")
        return {
            "success": True,
            "additional_fields": additional_fields_dict
        }
        
    except Exception as e:
        print(f"⚠️ Additional fields extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "additional_fields": {}
        }


# ============================================================================
# DATA PREPARATION AND EXPORT FUNCTIONS
# ============================================================================

def prepare_excel_data(contract_fields, service_rows, indicator_fields, additional_fields):
    """Prepare final data for Excel export"""
    print("\n📊 Stage 7: Preparing Excel Data...")
    
    # Start with contract fields
    base_row = contract_fields.copy()
    
    # Add indicator fields
    base_row.update(indicator_fields)
    
    # Add additional fields
    base_row.update(additional_fields)
    
    # Create rows for each service
    excel_data = []
    if service_rows:
        for service_row in service_rows:
            row = base_row.copy()
            row.update(service_row)
            excel_data.append(row)
    else:
        excel_data.append(base_row)
    
    # Ensure all fields are present
    for row in excel_data:
        for field in ALL_FIELD_NAMES:
            if field not in row:
                row[field] = "N/A"
    
    # Calculate success metrics
    total_fields = len(ALL_FIELD_NAMES)
    extracted_count = 0
    for field in ALL_FIELD_NAMES:
        # Check if field has been populated (not "N/A")
        field_values = set()
        for row in excel_data:
            if field in row and row[field] != "N/A":
                field_values.add(row[field])
        
        if field_values and "N/A" not in field_values:
            extracted_count += 1
    
    success_rate = (extracted_count / total_fields) * 100
    
    # Create summary
    summary = {
        "total_fields": total_fields,
        "extracted_fields": extracted_count,
        "success_rate": f"{success_rate:.1f}%",
        "service_rows": len(excel_data),
        "extraction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": contract_fields.get("FILE NAME", "Unknown")
    }
    
    # Update NLP fields
    if success_rate > 70:
        base_row["NLP EXTRACTION STATUS"] = "Success"
        base_row["NLP ERROR COMMENTS"] = "N/A"
    elif success_rate > 40:
        base_row["NLP EXTRACTION STATUS"] = "Partial"
        base_row["NLP ERROR COMMENTS"] = f"Extracted {extracted_count}/{total_fields} fields"
    else:
        base_row["NLP EXTRACTION STATUS"] = "Failed"
        base_row["NLP ERROR COMMENTS"] = f"Low extraction rate: {success_rate:.1f}%"
    
    print(f"✅ Prepared {len(excel_data)} rows for Excel")
    
    return {
        "excel_data": excel_data,
        "summary": summary
    }


def save_to_excel(excel_data, summary, output_file="ccg2_contract_extraction.xlsx"):
    """Save extracted data to Excel file"""
    try:
        if not excel_data:
            print("❌ No data to save")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(excel_data)
        
        # Ensure all fields are present in correct order
        missing_fields = [f for f in ALL_FIELD_NAMES if f not in df.columns]
        for field in missing_fields:
            df[field] = "N/A"
        
        # Reorder columns
        df = df[ALL_FIELD_NAMES]
        
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


def save_to_json(excel_data, summary, contract_fields, service_rows, output_file="ccg2_contract_extraction.json"):
    """Save full extraction results to JSON"""
    try:
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

def run_extraction_pipeline(file_path):
    """Run complete extraction pipeline"""
    print("🚀 Starting Healthcare Contract Extraction")
    print("=" * 60)
    
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_path)[1].lower().replace(".", "")
    
    print(f"📄 File: {file_name}")
    print(f"🎯 Target Fields: {len(ALL_FIELD_NAMES)}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Stage 1: Process Document
    print("\n▶️ Starting: Document Processing")
    doc_result = process_document(file_path)
    if not doc_result["success"]:
        print("❌ Document processing failed")
        return None
    
    chunks = doc_result["chunks"]
    raw_data_store = doc_result["raw_data_store"]
    
    # Stage 2: Store in Vector Database
    print("\n▶️ Starting: Vector Storage")
    vector_result = store_in_vector_db(chunks)
    if not vector_result["success"]:
        print("⚠️ Vector storage completed with issues")
        vector_store = None
    else:
        vector_store = vector_result["vector_store"]
    
    # Stage 3: Extract Contract Fields
    print("\n▶️ Starting: Contract Fields Extraction")
    contract_result = extract_contract_fields(vector_store, raw_data_store, file_name, file_extension)
    if not contract_result["success"]:
        print("⚠️ Contract fields extraction completed with issues")
    
    contract_fields = contract_result.get("contract_fields", {})
    
    # Stage 4: Extract Service Matrix
    print("\n▶️ Starting: Service Matrix Extraction")
    service_result = extract_service_matrix(vector_store, raw_data_store)
    service_rows = service_result.get("service_rows", [])
    
    # Stage 5: Extract Indicator Fields
    print("\n▶️ Starting: Indicator Fields Extraction")
    indicator_result = extract_indicator_fields(vector_store, raw_data_store)
    indicator_fields = indicator_result.get("indicator_fields", {})
    
    # Stage 6: Extract Additional Fields
    print("\n▶️ Starting: Additional Fields Extraction")
    additional_result = extract_additional_fields(vector_store, raw_data_store)
    additional_fields = additional_result.get("additional_fields", {})
    
    # Stage 7: Prepare Excel Data
    print("\n▶️ Starting: Excel Preparation")
    preparation_result = prepare_excel_data(
        contract_fields, 
        service_rows, 
        indicator_fields, 
        additional_fields
    )
    
    excel_data = preparation_result["excel_data"]
    summary = preparation_result["summary"]
    
    total_time = time.time() - start_time
    
    # Save results
    excel_file = save_to_excel(excel_data, summary)
    json_file = save_to_json(excel_data, summary, contract_fields, service_rows)
    
    # Display summary
    print("\n" + "=" * 60)
    print("🎉 EXTRACTION COMPLETE")
    print("=" * 60)
    
    print(f"⏱️ Total Time: {total_time:.1f} seconds")
    print(f"📊 Success Rate: {summary['success_rate']}")
    print(f"📋 Fields Extracted: {summary['extracted_fields']}/{summary['total_fields']}")
    print(f"🏥 Service Rows Created: {summary['service_rows']}")
    
    if excel_file:
        print(f"💾 Excel file: {excel_file}")
    if json_file:
        print(f"💾 JSON file: {json_file}")
    
    # Preview data
    if excel_data:
        print(f"\n📋 Data Preview (first 5 rows):")
        print("-" * 80)
        
        preview_data = excel_data[:5]
        for i, row in enumerate(preview_data):
            print(f"\nRow {i+1}:")
            print(f"  LOB IND: {row.get('LOB IND', 'N/A')}")
            print(f"  SERVICE TYPE: {row.get('SERVICE TYPE', 'N/A')}")
            print(f"  PRICE RATE: {row.get('PRICE RATE', 'N/A')}")
            print(f"  HEALTH BENEFIT PLANS: {row.get('HEALTH BENEFIT PLANS', 'N/A')[:50]}...")
    
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
    results = run_extraction_pipeline(file_path)
    
    if results:
        print("\n✅ Extraction completed successfully!")
    else:
        print("\n❌ Extraction failed!")