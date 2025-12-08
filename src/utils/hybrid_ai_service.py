"""
LedgerX - Hybrid AI Service
Low-cost AI enhancements: PO matching, GL validation, Exception explanations
Total cost: $0.20/month
"""

import os
import logging
from typing import Dict, List, Optional
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

# OpenAI for embeddings (cheap) and GPT-4 for exceptions only
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

# Small embedding model for local use (free)
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, runs locally
    EMBEDDINGS_ENABLED = True
    logger.info("[HYBRID AI] ✅ Local embeddings model loaded")
except:
    EMBEDDINGS_ENABLED = False
    logger.warning("[HYBRID AI] ⚠️ Embeddings model not available")

# -------------------------------------------------------------------
# 1. PO MATCHING (Embeddings API - $0.10/month)
# -------------------------------------------------------------------

# Sample PO database (in production, load from Cloud SQL)
PO_DATABASE = [
    {"po_number": "PO-2024-001", "vendor": "Global Materials Inc", "amount": 18900.50, "items": "Industrial Solvents"},
    {"po_number": "PO-2024-002", "vendor": "Tech Consultants Co", "amount": 5120.00, "items": "Consulting Services"},
    {"po_number": "PO-2024-003", "vendor": "Apex Supplies Ltd", "amount": 421.15, "items": "Office Supplies"},
    {"po_number": "PO-2024-004", "vendor": "ACME Manufacturing", "amount": 3500.20, "items": "Manufacturing Parts"},
]

def fuzzy_match_po(invoice_vendor: str, invoice_amount: float, invoice_items: str = "") -> Dict:
    """
    Use embeddings for fuzzy PO matching
    Cost: ~$0.10/month for 1000 invoices (using local model - FREE)
    """
    if not EMBEDDINGS_ENABLED:
        return {"matched": False, "confidence": 0, "po_number": None}
    
    try:
        # Create query text
        query = f"{invoice_vendor} {invoice_amount} {invoice_items}"
        
        # Generate embedding for query
        query_embedding = embedding_model.encode([query])
        
        # Generate embeddings for PO database
        po_texts = [f"{po['vendor']} {po['amount']} {po['items']}" for po in PO_DATABASE]
        po_embeddings = embedding_model.encode(po_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, po_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Match if similarity > 0.7
        if best_score > 0.7:
            matched_po = PO_DATABASE[best_idx]
            return {
                "matched": True,
                "confidence": float(best_score),
                "po_number": matched_po["po_number"],
                "po_vendor": matched_po["vendor"],
                "po_amount": matched_po["amount"],
                "amount_match": abs(invoice_amount - matched_po["amount"]) < 100,
                "discrepancy": abs(invoice_amount - matched_po["amount"])
            }
        else:
            return {
                "matched": False,
                "confidence": float(best_score),
                "po_number": None,
                "reason": "No PO found with >70% similarity"
            }
            
    except Exception as e:
        logger.error(f"[PO MATCH] Error: {e}")
        return {"matched": False, "confidence": 0, "error": str(e)}


# -------------------------------------------------------------------
# 2. GL CODE VALIDATION (Fine-tuned BERT - One-time training, then FREE)
# -------------------------------------------------------------------

# Chart of Accounts (COA) - in production, load from database
CHART_OF_ACCOUNTS = {
    "400-5110": {"name": "Raw Materials - Industrial", "category": "COGS", "typical_vendors": ["Global Materials", "Industrial Supply"]},
    "500-2110": {"name": "Consulting Services", "category": "Operating Expenses", "typical_vendors": ["Tech Consultants", "Advisory"]},
    "600-9870": {"name": "Shipping & Handling", "category": "Operating Expenses", "typical_vendors": ["UPS", "FedEx", "Logistics"]},
    "600-1020": {"name": "Office Supplies", "category": "Operating Expenses", "typical_vendors": ["Office Depot", "Staples", "Apex"]},
    "400-3020": {"name": "Manufacturing Parts", "category": "COGS", "typical_vendors": ["ACME", "Manufacturing"]},
    "400-2010": {"name": "Raw Materials - General", "category": "COGS", "typical_vendors": ["Materials", "Supply"]},
    "600-1010": {"name": "Office Paper & Printing", "category": "Operating Expenses", "typical_vendors": ["Office", "Paper"]},
}

def validate_gl_code(gl_code: str, line_item_description: str, vendor: str) -> Dict:
    """
    Validate GL code using local semantic matching
    In production: Use fine-tuned BERT model (one-time training, then free)
    Current: Rule-based with embeddings
    """
    if not EMBEDDINGS_ENABLED:
        return {"valid": True, "confidence": 0.5, "gl_code": gl_code}
    
    if gl_code not in CHART_OF_ACCOUNTS:
        return {
            "valid": False,
            "confidence": 0.0,
            "reason": f"GL code {gl_code} not in chart of accounts",
            "suggested_codes": list(CHART_OF_ACCOUNTS.keys())[:3]
        }
    
    try:
        account_info = CHART_OF_ACCOUNTS[gl_code]
        
        # Create query from line item and vendor
        query = f"{line_item_description} {vendor}"
        query_embedding = embedding_model.encode([query])
        
        # Create embedding for GL account
        account_text = f"{account_info['name']} {account_info['category']} {' '.join(account_info['typical_vendors'])}"
        account_embedding = embedding_model.encode([account_text])
        
        # Calculate similarity
        similarity = cosine_similarity(query_embedding, account_embedding)[0][0]
        
        return {
            "valid": similarity > 0.5,
            "confidence": float(similarity),
            "gl_code": gl_code,
            "account_name": account_info["name"],
            "category": account_info["category"],
            "match_score": float(similarity)
        }
        
    except Exception as e:
        logger.error(f"[GL VALIDATION] Error: {e}")
        return {"valid": True, "confidence": 0.5, "gl_code": gl_code, "error": str(e)}


# -------------------------------------------------------------------
# 3. EXCEPTION EXPLANATIONS (GPT-4 - Only for exceptions, $0.10/month)
# -------------------------------------------------------------------

def explain_exception(invoice_data: Dict, anomaly_type: str, details: Dict) -> str:
    """
    Use GPT-4 to explain exceptions in natural language
    Only called for flagged invoices (~5% of total)
    Cost: $0.10/month for ~50 exceptions
    """
    if not OPENAI_API_KEY:
        return f"Exception detected: {anomaly_type}. Manual review recommended."
    
    try:
        prompt = f"""You are a financial auditor. Explain this invoice exception in 1-2 sentences for an accountant.

Invoice Details:
- Vendor: {invoice_data.get('vendor_name', 'Unknown')}
- Amount: ${invoice_data.get('total_amount', 0):.2f}
- Anomaly: {anomaly_type}

Technical Details:
{details}

Provide a clear, actionable explanation:"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        explanation = response.choices[0].message.content.strip()
        logger.info(f"[GPT-4] ✅ Exception explained (cost: ~$0.002)")
        return explanation
        
    except Exception as e:
        logger.error(f"[GPT-4] Error: {e}")
        return f"Exception: {anomaly_type}. Review required."


# -------------------------------------------------------------------
# ENHANCED INVOICE PROCESSING
# -------------------------------------------------------------------

def process_invoice_with_hybrid_ai(invoice_data: Dict) -> Dict:
    """
    Complete invoice processing with hybrid AI enhancements
    
    Returns enhanced results with:
    - ML predictions (quality + failure)
    - PO matching results
    - GL code validation
    - Natural language explanations (if exception)
    """
    
    # Step 1: Get ML predictions (your existing models)
    from .inference_service import predict_invoice
    ml_results = predict_invoice(invoice_data)
    
    # Step 2: PO Matching (embeddings - cheap/free)
    po_match = fuzzy_match_po(
        invoice_vendor=invoice_data.get("vendor_name", ""),
        invoice_amount=invoice_data.get("total_amount", 0),
        invoice_items=invoice_data.get("line_items_text", "")
    )
    
    # Step 3: GL Code Validation (for each line item)
    gl_validations = []
    line_items = invoice_data.get("line_items", [])
    for item in line_items:
        gl_result = validate_gl_code(
            gl_code=item.get("gl_code", ""),
            line_item_description=item.get("description", ""),
            vendor=invoice_data.get("vendor_name", "")
        )
        gl_validations.append(gl_result)
    
    # Step 4: Generate explanation if exception
    explanation = None
    if ml_results["quality_bad"] == 1 or ml_results["failure_risk"] == 1:
        explanation = explain_exception(
            invoice_data=invoice_data,
            anomaly_type="Quality Issue" if ml_results["quality_bad"] == 1 else "Failure Risk",
            details={
                "quality_prob": ml_results["quality_probability"],
                "failure_prob": ml_results["failure_probability"],
                "warnings": ml_results["warnings"]
            }
        )
    
    # -------------------------------------------------------------------
    # COMPLETE RESPONSE
    # -------------------------------------------------------------------
    return {
        # ML predictions
        "quality": {
            "prediction": "bad" if ml_results["quality_bad"] == 1 else "good",
            "confidence": ml_results["quality_probability"],
            "probability": ml_results["quality_probability"]
        },
        "failure": {
            "prediction": "risk" if ml_results["failure_risk"] == 1 else "safe",
            "confidence": ml_results["failure_probability"],
            "probability": ml_results["failure_probability"]
        },
        
        # Hybrid AI enhancements
        "po_matching": po_match,
        "gl_validation": {
            "validated_codes": len(gl_validations),
            "all_valid": all(g.get("valid", False) for g in gl_validations),
            "results": gl_validations
        },
        "ai_explanation": explanation,
        
        # Routing decision
        "recommendation": "APPROVE" if (ml_results["quality_bad"] == 0 and 
                                        ml_results["failure_risk"] == 0 and 
                                        po_match.get("matched", False)) else "REVIEW",
        
        "warnings": ml_results["warnings"],
        "processing_mode": "hybrid_ai"
    }


# -------------------------------------------------------------------
# USAGE TRACKING
# -------------------------------------------------------------------

def track_hybrid_ai_usage():
    """Track usage for cost monitoring"""
    # In production, save to billing_usage table
    usage = {
        "embeddings_calls": 0,  # Free (local model)
        "gpt4_calls": 0,  # Only exceptions (~5%)
        "estimated_monthly_cost": 0.20
    }
    return usage