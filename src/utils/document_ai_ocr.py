"""
LedgerX - Google Document AI OCR Module
========================================

Replaces Tesseract with Document AI for 95%+ accuracy
"""

import os
import logging
from typing import Dict, Any
from datetime import datetime
from google.cloud import documentai_v1 as documentai

logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = "671429123152"
LOCATION = "us"
PROCESSOR_ID = "1903e3a537160b1f"

class DocumentAIProcessor:
    """Google Document AI invoice processor"""
    
    def __init__(self):
        """Initialize Document AI client"""
        self.client = documentai.DocumentProcessorServiceClient()
        self.processor_name = self.client.processor_path(
            PROJECT_ID, LOCATION, PROCESSOR_ID
        )
        logger.info(f"[DOC-AI] ✅ Initialized processor: {self.processor_name}")
    
    def process_invoice(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """
        Process invoice image with Document AI
        
        Args:
            image_bytes: Image file bytes
            mime_type: image/jpeg, image/png, or application/pdf
            
        Returns:
            Structured invoice data dictionary
        """
        logger.info(f"[DOC-AI] Processing invoice ({len(image_bytes)} bytes, {mime_type})")
        
        # Create document request
        raw_document = documentai.RawDocument(
            content=image_bytes,
            mime_type=mime_type
        )
        
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document
        )
        
        # Process document
        result = self.client.process_document(request=request)
        document = result.document
        
        logger.info(f"[DOC-AI] ✅ Extracted {len(document.entities)} entities")
        
        # Extract structured data
        invoice_data = self._extract_invoice_fields(document)
        
        logger.info(f"[DOC-AI] Invoice: {invoice_data.get('invoice_number')} | "
                   f"Vendor: {invoice_data.get('vendor_name')} | "
                   f"Amount: {invoice_data.get('total_amount')}")
        
        return invoice_data
    
    def _extract_invoice_fields(self, document) -> Dict[str, Any]:
        """Extract invoice fields from Document AI response"""
        
        # Initialize with defaults
        invoice_data = {
            "invoice_number": "",
            "vendor_name": "",
            "total_amount": 0.0,
            "currency": "USD",
            "invoice_date": "",
            "subtotal": 0.0,
            "tax_amount": 0.0,
            "tax_rate": 0.0,
            "discount_amount": 0.0,
            "line_items": [],
            "ocr_text": document.text,
            "ocr_confidence": 0.0
        }
        
        # Extract entities
        for entity in document.entities:
            entity_type = entity.type_
            entity_text = entity.mention_text.strip()
            confidence = entity.confidence
            
            logger.debug(f"[DOC-AI] Entity: {entity_type} = {entity_text} (conf: {confidence:.2f})")
            
            # Map Document AI fields to LedgerX schema
            if entity_type == "invoice_id":
                invoice_data["invoice_number"] = entity_text
            elif entity_type == "supplier_name":
                invoice_data["vendor_name"] = entity_text
            elif entity_type == "total_amount":
                invoice_data["total_amount"] = self._parse_amount(entity_text)
            elif entity_type == "currency":
                invoice_data["currency"] = entity_text[:3].upper()  # Normalize to 3-letter code
            elif entity_type == "invoice_date":
                invoice_data["invoice_date"] = self._parse_date(entity_text)
            elif entity_type == "net_amount":
                invoice_data["subtotal"] = self._parse_amount(entity_text)
            elif entity_type == "total_tax_amount":
                invoice_data["tax_amount"] = self._parse_amount(entity_text)
        
        # Extract line items
        for entity in document.entities:
            if entity.type_ == "line_item":
                line_item = {}
                for prop in entity.properties:
                    if prop.type_ == "line_item/description":
                        line_item["description"] = prop.mention_text
                    elif prop.type_ == "line_item/amount":
                        line_item["amount"] = self._parse_amount(prop.mention_text)
                    elif prop.type_ == "line_item/quantity":
                        try:
                            line_item["quantity"] = float(prop.mention_text)
                        except:
                            line_item["quantity"] = 1.0
                    elif prop.type_ == "line_item/unit_price":
                        line_item["unit_price"] = self._parse_amount(prop.mention_text)
                
                if line_item:
                    invoice_data["line_items"].append(line_item)
        
        # Calculate OCR confidence (average of all entities)
        invoice_data["ocr_confidence"] = self._calculate_avg_confidence(document)
        
        # Calculate additional features for ML model
        invoice_data["blur_score"] = 85.0  # Document AI handles quality internally
        invoice_data["contrast_score"] = 80.0
        invoice_data["vendor_freq"] = 0.05  # Default, will be updated later
        invoice_data["file_size_kb"] = 0.0  # Will be set by caller
        
        return invoice_data
    
    def _parse_amount(self, text: str) -> float:
        """Parse amount string to float"""
        import re
        # Remove currency symbols, spaces, and commas
        cleaned = re.sub(r'[€$£,\s]', '', text)
        try:
            return float(cleaned)
        except:
            return 0.0
    
    def _parse_date(self, text: str) -> str:
        """Parse date to YYYY-MM-DD format"""
        try:
            # Try multiple date formats
            for fmt in ['%d-%b-%Y', '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']:
                try:
                    dt = datetime.strptime(text, fmt)
                    return dt.strftime('%Y-%m-%d')
                except:
                    continue
            
            # If no format matches, return as-is
            return text
        except:
            return datetime.now().strftime('%Y-%m-%d')
    
    def _calculate_avg_confidence(self, document) -> float:
        """Calculate average confidence across all entities"""
        confidences = [
            entity.confidence 
            for entity in document.entities 
            if entity.confidence > 0
        ]
        
        if not confidences:
            return 0.85  # Default if no confidence scores
        
        return sum(confidences) / len(confidences)


# Global processor instance (initialized once)
_processor = None

def get_processor() -> DocumentAIProcessor:
    """Get global DocumentAI processor instance"""
    global _processor
    if _processor is None:
        _processor = DocumentAIProcessor()
    return _processor