import os
from pathlib import Path
import json
import re
import time
import base64
import shutil
import pandas as pd
from typing import Dict, List, Tuple, Any
from io import StringIO

# OpenAI imports
from openai import OpenAI
import faiss
import numpy as np
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import pdfplumber
from PIL import Image
import io

# --- GLOBAL SETUP & CONSTANTS ---
# Initialize OpenAI client
OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Model Selection - Optimized for OpenAI ---
TEXT_LLM_MODEL = "gpt-4.1-mini"  # Fast and accurate
# TEXT_LLM_MODEL = "gpt-4.1"  # More accurate but slower
MULTIMODAL_MODEL = "gpt-4o"  # Vision model
EMBEDDING_MODEL = "text-embedding-3-small"  # Fast and efficient

# --- CCG2.0 Fields (58 Fields from Excel) ---
CCG2_FIELDS_SCHEMA = {
    # === CONTRACT LEVEL FIELDS ===
    "CONTRACT_IDENTIFICATION": [
        "CIS CTRCT ID", "ATTACHMENT ID", "FILE NAME", "FILE EXTENSION",
        "CIS TYPE", "CIS TYPE DESCRIPTION", "PROVIDER NAME", "NPI", 
        "TAX ID", "PROVZIPCODE", "TAXONOMYCODE", "EFFECTIVE FROM DATE",
        "EFFECTIVE TO DATE", "PLACEOFSERV", "PROVIDER SPECIALITY"
    ],
    
    # === SERVICE LEVEL FIELDS === 
    "SERVICE_DETAILS": [
        "SERVICE TYPE", "SERVICE DESC", "SERVICES", "AGE GROUP", "CODES",
        "GROUPER", "PRICE RATE", "REVENUE CD IND", "DRG CD IND", "CPT IND",
        "HCPCS IND", "ICD CD IND", "DIAGNOSIS CD IND", "MODIFIER CD IND",
        "GROUPER IND", "APC IND", "EXCLUSION IND", "MSR IND", "BILETRAL PROCEDURE IND",
        "EXCLUDE FROM TRANSFER IND", "EXCLUDE FROM STOPLOSS IND"
    ],
    
    # === DISCHARGE & PAYMENT FIELDS ===
    "DISCHARGE_PAYMENT": [
        "DISCHARGESTATUSCODE", "ALOSGLOS", "TRANSFER RATE", "APPLIEDTRANSFERCASE",
        "ISTHRESHOLD", "ISCAPAMOUNT", "REIMBURSEMENT AMT", "REIMBURSEMENT RATE",
        "REIMBURSEMENT METHODOLOGY", "MULTIPLERMETHODS", "METHOD OF PAYMENT",
        "HEALTH BENEFIT PLANS", "ADDITIONAL NOTES", "OTHER FLAT FEE", "SURG FLAT FEE",
        "AND OR OPERATOR", "OPERATOR CODE TYPE"
    ],
    
    # === LANGUAGE CLAUSE FIELDS ===
    "LANGUAGE_CLAUSES": [
        "READMISSION LANG IND", "READMISSION LANG TIMEFRAME", "READMISSION LANG",
        "LABS LANG IND", "LABS LANG CODES", "LABS LANG", "MODIFICATION LANG IND",
        "MODIFICATION LANG NOTICE PERIOD", "MODIFICATION LANG", "NEW PROD LANG IND",
        "NEW PROD LANG NOTICE PERIOD", "NEW PROD LANG", "INTRA FACILITY TRANSFER LANG IND",
        "INTRA FACILITY TRANSFER LANG TIMEFRAME", "INTRA FACILITY TRANSFER LANG",
        "POST DISCHG TESTING LANG IND", "POST DISCHG TESTING LANG TIMEFRAME", "POST DISCHG TESTING LANG"
    ],
    
    # === NLP PROCESSING FIELDS ===
    "NLP_FIELDS": [
        "NLP EXTRACTION STATUS", "NLP USER ID", "NLP PROCESS TIMESTAMP", "NLP ERROR COMMENTS"
    ]
}

ALL_FIELD_NAMES = [field for category in CCG2_FIELDS_SCHEMA.values() for field in category]

# --- Confidence Rules ---
CONFIDENCE_RULES = {
    "HIGH_CONFIDENCE_TEXT": [
        "PROVIDER NAME", "NPI", "TAX ID", "PROVZIPCODE", "EFFECTIVE FROM DATE",
        "EFFECTIVE TO DATE", "FILE NAME", "FILE EXTENSION", "CIS TYPE DESCRIPTION"
    ],
    "MEDIUM_CONFIDENCE_TABULAR": [
        "CODES", "PRICE RATE", "REIMBURSEMENT RATE", "SERVICE TYPE", "SERVICES",
        "AGE GROUP", "REIMBURSEMENT AMT", "OTHER FLAT FEE", "SURG FLAT FEE"
    ],
    "LOW_CONFIDENCE_COMPLEX": [
        "REIMBURSEMENT METHODOLOGY", "APPLIEDTRANSFERCASE", "ALOSGLOS",
        "TRANSFER RATE", "MULTIPLERMETHODS", "HEALTH BENEFIT PLANS"
    ]
}

# --- Helper Functions ---
def _get_image_base64(image_path_str):
    """Convert image to Base64 for multimodal processing"""
    if os.path.exists(image_path_str):
        try:
            with open(image_path_str, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception:
            return None
    return None

def safe_string_convert(value: Any) -> str:
    """Safely convert any value to string"""
    if value is None:
        return "N/A"
    elif isinstance(value, (list, dict)):
        return json.dumps(value)
    elif isinstance(value, str):
        return value.strip() if value.strip() else "N/A"
    else:
        return str(value) if str(value).strip() else "N/A"

def calculate_confidence(field_name, extracted_value):
    """Calculate confidence score for extracted field"""
    if extracted_value in ["N/A", "", None]:
        return 0.0
    
    if field_name in CONFIDENCE_RULES["HIGH_CONFIDENCE_TEXT"]:
        base_score = 0.8
    elif field_name in CONFIDENCE_RULES["MEDIUM_CONFIDENCE_TABULAR"]:
        base_score = 0.5
    else:
        base_score = 0.3
    
    # Adjust based on value quality
    value = str(extracted_value)
    if len(value) > 2 and value not in ["N/A", "None", "null"]:
        return min(1.0, base_score + 0.2)
    return base_score

def get_openai_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI"""
    try:
        response = OPENAI_CLIENT.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return [0.0] * 1536  # text-embedding-3-small dimension

# --- Optimized Healthcare Contract Extractor with FAISS ---
class HealthcareContractExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.extracted_elements = [] 
        self.raw_data_store = {}     
        self.image_dir = "extracted_images"
        self.faiss_index = None
        self.index_to_id = {}
        
        # Store all extracted images for scanned PDF fallback
        self.all_extracted_images = {}  # {page_number: [image_paths]}
        
        # Extraction results with confidence
        self.extraction_results = {
            "fields": {},
            "confidence_scores": {},
            "extraction_method": {},
            "multimodal_corrections": 0
        }
        
        # Clean setup for Databricks
        # Use /tmp for temporary files in Databricks
        self.temp_dir = "/tmp/contract_extraction"
        self.image_dir = os.path.join(self.temp_dir, "extracted_images")
        
        # Clean previous runs
        for dir_path in [self.temp_dir, self.image_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
        
        print("✅ FAISS vector database ready.")

    def partition_and_chunk(self):
        """
        Stage 1: Extract text and images - optimized for scanned PDFs
        """
        if not os.path.exists(self.file_path):
            print(f"❌ File not found: {self.file_path}")
            return
        
        print("--- Stage 1: PDF Processing & Text Extraction ---")
        
        try:
            # Use hi_res strategy for scanned PDFs, auto for digital
            strategy = "hi_res"  # Better for scanned images
            
            self.extracted_elements = partition_pdf(
                filename=self.file_path,
                strategy=strategy,
                infer_table_structure=True,
                extract_images_in_pdf=True,
                extract_image_block_types=["Image", "Table"],
                pdf_extract_images_path=self.image_dir,
                extract_tables=True
            )
            print(f"✅ Extracted {len(self.extracted_elements)} elements")
            
        except Exception as e:
            print(f"❌ PDF processing failed: {e}")
            return

        # Catalog extracted images for multimodal fallback
        self._catalog_extracted_images()
        
        # Check if document is mostly scanned (low text content)
        text_elements = [e for e in self.extracted_elements if hasattr(e, 'text') and e.text and e.text.strip()]
        total_text_length = sum(len(e.text) for e in text_elements)
        
        if total_text_length < 1000:  # Likely scanned PDF
            print("📄 Document appears to be scanned PDF - will rely more on multimodal")
            self.is_scanned_pdf = True
        else:
            self.is_scanned_pdf = False
        
        # Prepare text chunks for RAG
        print(f"📝 Preparing {len(text_elements)} text elements for chunking...")
        
        self.chunks = chunk_by_title(
            elements=text_elements,
            max_characters=1000,  # Smaller chunks for faster processing
            new_after_n_chars=700,
            combine_text_under_n_chars=150
        )
        print(f"✅ Created {len(self.chunks)} text chunks for RAG.")
    
    def _catalog_extracted_images(self):
        """Catalog all images for multimodal fallback"""
        print("📸 Cataloging extracted images for multimodal fallback...")
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(self.image_dir).glob(ext))
        
        self.all_extracted_images = {}
        for img_path in image_files:
            page_num = 1
            numbers = re.findall(r'\d+', img_path.name)
            if numbers:
                page_num = int(numbers[0])
            
            if page_num not in self.all_extracted_images:
                self.all_extracted_images[page_num] = []
            self.all_extracted_images[page_num].append(str(img_path))
        
        total_images = sum(len(imgs) for imgs in self.all_extracted_images.values())
        print(f"✅ Cataloged {total_images} images across {len(self.all_extracted_images)} pages")
    
    def store_all_content(self):
        """Stage 2: Store all content in FAISS vector database"""
        if not hasattr(self, 'chunks') or not self.chunks:
            print("❌ No chunks to store.")
            return 0
            
        print("--- Stage 2: Creating FAISS Index ---")
        raw_data_id_counter = 0
        documents = []
        
        for chunk in self.chunks:
            raw_id = f"raw_data_{raw_data_id_counter}"
            
            content_to_store = chunk.text
            page_num = getattr(chunk.metadata, 'page_number', 1)
            
            self.raw_data_store[raw_id] = {
                "content": content_to_store,
                "page": page_num
            } 
            
            documents.append(content_to_store)
            self.index_to_id[raw_data_id_counter] = raw_id
            raw_data_id_counter += 1

        # Create FAISS index
        embeddings = []
        for doc in documents:
            embedding = get_openai_embedding(doc)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        # Create FAISS index (IndexFlatL2 for accuracy, IndexIVFFlat for speed with large datasets)
        if len(documents) > 1000:
            # For large datasets, use IndexIVFFlat
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.faiss_index.train(embeddings_array)
            self.faiss_index.add(embeddings_array)
            self.faiss_index.nprobe = 10  # number of clusters to visit
        else:
            # For smaller datasets, use simple IndexFlatL2
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings_array)
        
        print(f"✅ Stored {len(documents)} chunks in FAISS vector database.")
        return len(documents)
    
    def _search_faiss(self, query_text: str, k: int = 8):
        """Search FAISS index for relevant documents"""
        if self.faiss_index is None:
            return []
        
        query_embedding = get_openai_embedding(query_text)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search
        if k > self.faiss_index.ntotal:
            k = self.faiss_index.ntotal
        
        distances, indices = self.faiss_index.search(query_vector, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:  # Valid index
                raw_id = self.index_to_id.get(idx)
                if raw_id:
                    raw_data = self.raw_data_store.get(raw_id)
                    if raw_data:
                        results.append({
                            'raw_id': raw_id,
                            'content': raw_data['content'],
                            'page': raw_data['page'],
                            'distance': float(distance)
                        })
        
        return results

    def _get_relevant_context_for_fields(self):
        """
        Stage 3: Smart context retrieval optimized for CCG2 fields using FAISS
        """
        print("\n🔍 Stage 3: Retrieving Relevant Context (FAISS RAG)...")
        
        # Field-specific queries for better precision
        field_queries = {
            "provider_info": "provider name NPI tax identification number TIN",
            "service_codes": "CPT HCPCS ICD codes procedure codes service codes",
            "rates_payment": "rate price reimbursement payment fee schedule amount",
            "contract_terms": "effective date termination notice period term",
            "language_clauses": "readmission laboratory modification notice period timeframe"
        }
        
        all_context = []
        seen_raw_ids = set()
        
        for category, query in field_queries.items():
            try:
                results = self._search_faiss(query, k=8)
                
                for result in results:
                    raw_id = result['raw_id']
                    if raw_id not in seen_raw_ids:
                        all_context.append({
                            'raw_id': raw_id,
                            'content': result['content'],
                            'page': result['page'],
                            'category': category,
                            'distance': result['distance']
                        })
                        seen_raw_ids.add(raw_id)
            except Exception as e:
                print(f"⚠️  FAISS search failed for {category}: {e}")
                continue
        
        # Fallback with diverse content
        if not all_context and self.raw_data_store:
            print("⚠️  FAISS search limited, using diverse content...")
            for raw_id, data in list(self.raw_data_store.items())[:15]:
                all_context.append({
                    'raw_id': raw_id,
                    'content': data['content'],
                    'page': data['page'],
                    'category': 'general',
                    'distance': 0.0
                })
        
        # Sort by distance (lower is better)
        all_context.sort(key=lambda x: x['distance'])
        
        print(f"✅ Retrieved {len(all_context)} relevant context blocks.")
        return all_context

    def _format_context_for_llm(self, context_blocks):
        """Format context for LLM with optimal size"""
        formatted_context = ""
        total_chars = 0
        max_chars = 8000  # Reduced for faster processing
        
        for block in context_blocks:
            page = block['page']
            content = block['content']
            category = block.get('category', 'general')
            
            chunk_text = f"\n--- PAGE {page} ({category.upper()}) ---\n{content}\n"
            
            if total_chars + len(chunk_text) > max_chars:
                break
                
            formatted_context += chunk_text
            total_chars += len(chunk_text)
        
        print(f"📄 Context size: {len(formatted_context)} characters")
        return formatted_context

    def extract_fields_with_text_rag(self, context_blocks):
        """
        Stage 4a: Primary text extraction for all CCG2 fields using OpenAI
        """
        print("\n🤖 Stage 4a: Text-Only Extraction (OpenAI + FAISS RAG)...")
        
        context_string = self._format_context_for_llm(context_blocks)
        
        extraction_prompt = f"""EXTRACT ALL HEALTHCARE CONTRACT FIELDS:

You are a healthcare contract analyst. Extract these EXACT {len(ALL_FIELD_NAMES)} fields:

FIELD LIST: {json.dumps(ALL_FIELD_NAMES, indent=2)}

RULES:
- Use ONLY the field names above exactly as shown
- Return "N/A" for missing fields
- Keep values concise and accurate
- Extract from contract text below

CONTRACT CONTEXT:
{context_string}

OUTPUT JSON with all {len(ALL_FIELD_NAMES)} fields:"""

        try:
            start_time = time.time()
            response = OPENAI_CLIENT.chat.completions.create(
                model=TEXT_LLM_MODEL,
                messages=[{"role": "user", "content": extraction_prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4000
            )
            extraction_time = time.time() - start_time
            
            extracted_data = json.loads(response.choices[0].message.content)
            
            # Calculate confidence scores
            for field in ALL_FIELD_NAMES:
                value = extracted_data.get(field, "N/A")
                self.extraction_results["fields"][field] = safe_string_convert(value)
                self.extraction_results["confidence_scores"][field] = calculate_confidence(field, value)
                self.extraction_results["extraction_method"][field] = "text"
            
            print(f"✅ OpenAI text extraction completed in {extraction_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"❌ OpenAI text extraction failed: {e}")
            # Initialize all fields as failed
            for field in ALL_FIELD_NAMES:
                self.extraction_results["fields"][field] = "TEXT_EXTRACTION_FAILED"
                self.extraction_results["confidence_scores"][field] = 0.0
                self.extraction_results["extraction_method"][field] = "failed"
            return False

    def identify_low_confidence_fields(self):
        """Identify fields that need multimodal fallback"""
        low_confidence_fields = []
        
        for field_name, confidence in self.extraction_results["confidence_scores"].items():
            if confidence < 0.6:  # Threshold for multimodal fallback
                low_confidence_fields.append(field_name)
        
        print(f"🎯 {len(low_confidence_fields)} fields identified for multimodal fallback")
        return low_confidence_fields

    def extract_fields_with_multimodal(self, field_names):
        """
        Stage 4b: Multimodal fallback for low-confidence fields using GPT-4o Vision
        """
        if not field_names:
            return 0
            
        print(f"\n🖼️  Stage 4b: Multimodal Fallback with GPT-4o for {len(field_names)} fields...")
        
        corrections = 0
        
        # Process fields in batches for speed
        for i, field_name in enumerate(field_names[:10]):  # Limit to 10 fields for performance
            print(f"  🔍 Processing {field_name} with GPT-4o Vision...")
            
            # Find relevant pages for this field
            relevant_pages = self._find_relevant_pages_for_field(field_name)
            
            if not relevant_pages:
                continue
            
            # Try to extract from images on relevant pages
            field_value = self._extract_from_page_images_gpt4o(field_name, relevant_pages)
            
            if field_value and field_value != "NOT_FOUND" and field_value != "N/A":
                self.extraction_results["fields"][field_name] = field_value
                self.extraction_results["confidence_scores"][field_name] = 0.9  # High confidence for multimodal
                self.extraction_results["extraction_method"][field_name] = "multimodal"
                corrections += 1
                print(f"    ✅ Corrected {field_name}: {field_value}")
        
        return corrections

    def _find_relevant_pages_for_field(self, field_name):
        """Find pages that might contain this field"""
        relevant_pages = set()
        
        # Look for field mentions in text context
        for raw_id, data in self.raw_data_store.items():
            content = data['content'].lower()
            field_terms = [
                field_name.lower(),
                field_name.lower().replace('_', ' '),
                ' '.join(word for word in field_name.lower().split()[:2])  # First two words
            ]
            
            for term in field_terms:
                if term in content and len(term) > 3:
                    relevant_pages.add(data['page'])
                    break
        
        return list(relevant_pages)

    def _extract_from_page_images_gpt4o(self, field_name, page_numbers):
        """Extract field value from page images using GPT-4o Vision"""
        for page_num in page_numbers[:2]:  # Check first 2 relevant pages
            page_images = self.all_extracted_images.get(page_num, [])
            
            for img_path in page_images[:2]:  # Check first 2 images per page
                image_base64 = _get_image_base64(img_path)
                if not image_base64:
                    continue
                
                # Optimized prompt for GPT-4o Vision
                vision_prompt = f"""Analyze this healthcare contract page image.

SPECIFIC FIELD TO FIND: {field_name}

Look for this field in:
- Tables or structured data
- Forms or boxes
- Text near relevant sections
- Anywhere on the page

Return ONLY the value if found, otherwise "NOT_FOUND".
Be precise and concise."""

                try:
                    response = OPENAI_CLIENT.chat.completions.create(
                        model=MULTIMODAL_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": vision_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=500,
                        temperature=0.1
                    )
                    
                    result = response.choices[0].message.content.strip()
                    if result and result != "NOT_FOUND" and len(result) > 1:
                        return result
                        
                except Exception as e:
                    print(f"    ⚠️  GPT-4o Vision error: {e}")
                    continue
        
        return None

    def execute_complete_extraction(self):
        """
        Main method: Execute complete extraction pipeline
        """
        print("🚀 EXECUTING CCG2.0 CONTRACT EXTRACTION PIPELINE")
        print("="*60)
        print(f"🎯 Target Fields: {len(ALL_FIELD_NAMES)} CCG2.0 fields")
        print(f"📄 Document: {self.file_path}")
        print(f"🤖 Text Model: {TEXT_LLM_MODEL}")
        print(f"🖼️  Vision Model: {MULTIMODAL_MODEL}")
        print(f"🔤 Embedding Model: {EMBEDDING_MODEL}")
        print("="*60)
        
        start_total_time = time.time()
        
        # Stage 1-2: Process document
        self.partition_and_chunk()
        if not hasattr(self, 'chunks') or not self.chunks:
            print("❌ Document processing failed")
            return self._create_fallback_output()
        
        self.store_all_content()
        
        # Stage 3: Get context
        context_blocks = self._get_relevant_context_for_fields()
        if not context_blocks:
            print("❌ No context retrieved")
            return self._create_fallback_output()
        
        # Stage 4a: Primary text extraction
        text_success = self.extract_fields_with_text_rag(context_blocks)
        
        # Stage 4b: Multimodal fallback if needed
        if self.is_scanned_pdf or text_success:
            low_confidence_fields = self.identify_low_confidence_fields()
            multimodal_corrections = self.extract_fields_with_multimodal(low_confidence_fields)
            self.extraction_results["multimodal_corrections"] = multimodal_corrections
        else:
            multimodal_corrections = 0
        
        total_time = time.time() - start_total_time
        
        return self._generate_final_output(total_time)

    def _generate_final_output(self, total_time):
        """Generate final structured output"""
        successful_fields = sum(1 for field, value in self.extraction_results["fields"].items() 
                              if value not in ['N/A', 'TEXT_EXTRACTION_FAILED'] and len(str(value).strip()) > 1)
        
        success_rate = successful_fields / len(ALL_FIELD_NAMES)
        
        return {
            "extraction_summary": {
                "total_fields": len(ALL_FIELD_NAMES),
                "successfully_extracted": successful_fields,
                "success_rate": f"{success_rate:.1%}",
                "multimodal_corrections": self.extraction_results["multimodal_corrections"],
                "total_time_seconds": round(total_time, 2),
                "is_scanned_pdf": self.is_scanned_pdf,
                "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "models_used": {
                    "text": TEXT_LLM_MODEL,
                    "vision": MULTIMODAL_MODEL,
                    "embedding": EMBEDDING_MODEL
                }
            },
            "extracted_fields": [
                {
                    "attribute_name": field,
                    "value": self.extraction_results["fields"][field],
                    "confidence_score": self.extraction_results["confidence_scores"][field],
                    "extraction_method": self.extraction_results["extraction_method"][field],
                    "field_category": self._get_field_category(field)
                }
                for field in ALL_FIELD_NAMES
            ],
            "field_categories": {
                category: len(fields) 
                for category, fields in CCG2_FIELDS_SCHEMA.items()
            }
        }

    def _get_field_category(self, field_name):
        """Get the category of a field"""
        for category, fields in CCG2_FIELDS_SCHEMA.items():
            if field_name in fields:
                return category
        return "UNCATEGORIZED"

    def _create_fallback_output(self):
        """Create fallback output when extraction fails"""
        return {
            "extraction_summary": {
                "total_fields": len(ALL_FIELD_NAMES),
                "successfully_extracted": 0,
                "success_rate": "0%",
                "multimodal_corrections": 0,
                "error": "Extraction pipeline failed",
                "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "extracted_fields": [
                {
                    "attribute_name": field, 
                    "value": "EXTRACTION_FAILED",
                    "confidence_score": 0.0,
                    "extraction_method": "failed",
                    "field_category": self._get_field_category(field)
                }
                for field in ALL_FIELD_NAMES
            ]
        }


# --- Main Execution for Databricks ---
def run_extraction_on_databricks(file_path):
    """
    Main function to run extraction in Databricks environment
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Please ensure the PDF file exists in the Databricks file system.")
        return None
    
    print("🚀 CCG2.0 HEALTHCARE CONTRACT EXTRACTION SYSTEM")
    print("="*60)
    print(f"📄 Document: {file_path}")
    print(f"🎯 Target Fields: {len(ALL_FIELD_NAMES)} CCG2.0 fields")
    print(f"🤖 Text Model: {TEXT_LLM_MODEL}")
    print(f"🖼️  Vision Model: {MULTIMODAL_MODEL}")
    print(f"🔤 Embedding Model: {EMBEDDING_MODEL}")
    print("="*60)
    
    start_total_time = time.time()
    
    # Initialize and run extraction
    extractor = HealthcareContractExtractor(file_path)
    final_results = extractor.execute_complete_extraction()
    
    total_time = time.time() - start_total_time
    
    # Display results
    print("\n" + "="*80)
    print("🎉 EXTRACTION COMPLETE")
    print("="*80)
    
    summary = final_results["extraction_summary"]
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"📊 Success Rate: {summary['success_rate']}")
    print(f"📋 Fields Extracted: {summary['successfully_extracted']}/{summary['total_fields']}")
    print(f"🖼️  Multimodal Corrections: {summary['multimodal_corrections']}")
    print(f"🔍 Scanned PDF: {summary['is_scanned_pdf']}")
    print("="*80)
    
    # Save results to Databricks File System (DBFS)
    output_file = "/dbfs/tmp/ccg2_contract_extraction.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}")
    
    # Return results for further processing
    return final_results


if __name__ == "__main__":
    # Example usage in Databricks
    file_path = "/dbfs/tmp/sample-humana1.pdf"  # Adjust path for your Databricks setup
    
    # Set OpenAI API key from Databricks secrets
    import os
    from databricks import secrets
    
    # Get OpenAI API key from Databricks secrets
    try:
        OPENAI_API_KEY = secrets.get(scope="openai", key="api_key")
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    except:
        print("⚠️  Could not get OpenAI API key from secrets. Using environment variable.")
    
    results = run_extraction_on_databricks(file_path)
    
    if results:
        # Preview extracted fields by category
        print(f"\n🔍 EXTRACTED FIELDS BY CATEGORY:")
        print("-" * 70)
        
        successful_fields = [
            field for field in results["extracted_fields"] 
            if field['value'] not in ['N/A', 'TEXT_EXTRACTION_FAILED', 'EXTRACTION_FAILED'] 
            and len(field['value'].strip()) > 1
        ]
        
        # Group by category
        by_category = {}
        for field in successful_fields:
            category = field['field_category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(field)
        
        for category, fields in by_category.items():
            print(f"\n📁 {category.replace('_', ' ').title()}:")
            for field in fields[:5]:  # Show first 5 per category
                value_preview = field['value'][:50] + "..." if len(field['value']) > 50 else field['value']
                confidence = field['confidence_score']
                method = field['extraction_method']
                print(f"   • {field['attribute_name']:25} : {value_preview} (conf: {confidence:.1f}, {method})")
            
            if len(fields) > 5:
                print(f"   ... and {len(fields) - 5} more fields")