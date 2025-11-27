# src/stages/duplicate_detection.py

"""
LedgerX Duplicate Invoice Detection
Prevents duplicate payments through multi-strategy detection
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Tolerance for amount matching (5%)
AMOUNT_TOLERANCE = 0.05

# Time window for fuzzy matching (30 days)
TIME_WINDOW_DAYS = 30


class DuplicateError:
    """Represents a detected duplicate"""
    
    def __init__(self, 
                 strategy: str, 
                 confidence: float, 
                 match_data: Dict[str, Any],
                 original_invoice: Dict[str, Any]):
        self.strategy = strategy
        self.confidence = confidence
        self.match_data = match_data
        self.original_invoice = original_invoice
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy,
            'confidence': round(self.confidence, 4),
            'match_details': {
                'original_invoice_number': self.original_invoice.get('invoice_number', 'N/A'),
                'original_vendor': self.original_invoice.get('vendor_name', 'N/A'),
                'original_amount': self.original_invoice.get('total_amount', 0),
                'original_date': self.original_invoice.get('invoice_date', 'N/A'),
                'similarity_score': self.match_data.get('similarity_score', 1.0)
            }
        }


class DuplicateDetector:
    """Multi-strategy duplicate invoice detector"""
    
    def __init__(self, historical_invoices_path: Optional[Path] = None):
        """
        Initialize detector with historical invoice database
        
        Args:
            historical_invoices_path: Path to CSV file with historical invoices
        """
        self.historical_invoices_path = historical_invoices_path
        self.historical_df = None
        
        if historical_invoices_path and historical_invoices_path.exists():
            try:
                self.historical_df = pd.read_csv(historical_invoices_path)
                # Convert date to datetime
                if 'invoice_date' in self.historical_df.columns:
                    self.historical_df['invoice_date'] = pd.to_datetime(
                        self.historical_df['invoice_date'], 
                        errors='coerce'
                    )
                logger.info(f"✅ Loaded {len(self.historical_df)} historical invoices")
            except Exception as e:
                logger.error(f"❌ Failed to load historical invoices: {e}")
                self.historical_df = None
    
    def detect_duplicates(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all duplicate detection strategies
        
        Args:
            invoice_data: New invoice to check
            
        Returns:
            Detection result with duplicates found (if any)
        """
        duplicates: List[DuplicateError] = []
        
        # If no historical data, can't detect duplicates
        if self.historical_df is None or len(self.historical_df) == 0:
            logger.warning("⚠️ No historical invoice data available for duplicate detection")
            return {
                'is_duplicate': False,
                'duplicates_found': [],
                'duplicate_count': 0,
                'highest_confidence': 0.0,
                'strategies_used': ['exact', 'fuzzy_amount', 'similar_number']
            }
        
        # Strategy 1: Exact match
        exact_duplicates = self._detect_exact_match(invoice_data)
        duplicates.extend(exact_duplicates)
        
        # Strategy 2: Fuzzy amount match
        fuzzy_duplicates = self._detect_fuzzy_amount(invoice_data)
        duplicates.extend(fuzzy_duplicates)
        
        # Strategy 3: Similar invoice numbers (typos)
        typo_duplicates = self._detect_similar_invoice_number(invoice_data)
        duplicates.extend(typo_duplicates)
        
        # Calculate highest confidence
        highest_confidence = max([d.confidence for d in duplicates]) if duplicates else 0.0
        
        # Log results
        if duplicates:
            logger.warning(
                f"🚨 DUPLICATE DETECTED! Found {len(duplicates)} potential duplicate(s) "
                f"(confidence: {highest_confidence:.2%})"
            )
        else:
            logger.info("✅ No duplicates detected")
        
        return {
            'is_duplicate': len(duplicates) > 0,
            'duplicates_found': [d.to_dict() for d in duplicates],
            'duplicate_count': len(duplicates),
            'highest_confidence': highest_confidence,
            'strategies_used': ['exact', 'fuzzy_amount', 'similar_number']
        }
    
    def _detect_exact_match(self, invoice_data: Dict[str, Any]) -> List[DuplicateError]:
        """
        Strategy 1: Exact match on invoice_number + vendor + amount
        
        This catches true duplicates (resubmissions)
        """
        duplicates = []
        
        invoice_number = invoice_data.get('invoice_number', '')
        vendor_name = invoice_data.get('vendor_name', '')
        total_amount = float(invoice_data.get('total_amount', 0))
        
        if not invoice_number or not vendor_name:
            return duplicates
        
        # Find exact matches
        matches = self.historical_df[
            (self.historical_df['invoice_number'].astype(str) == str(invoice_number)) &
            (self.historical_df['vendor_name'].astype(str) == str(vendor_name)) &
            (self.historical_df['total_amount'].astype(float) == total_amount)
        ]
        
        for _, match in matches.iterrows():
            duplicates.append(DuplicateError(
                strategy='EXACT_MATCH',
                confidence=1.0,  # 100% confidence on exact match
                match_data={'similarity_score': 1.0},
                original_invoice=match.to_dict()
            ))
            logger.warning(
                f"🚨 EXACT MATCH: Invoice {invoice_number} already exists "
                f"(Vendor: {vendor_name}, Amount: ${total_amount})"
            )
        
        return duplicates
    
    def _detect_fuzzy_amount(self, invoice_data: Dict[str, Any]) -> List[DuplicateError]:
        """
        Strategy 2: Fuzzy amount match (±5% within 30 days, same vendor)
        
        This catches duplicates where amount was slightly adjusted
        """
        duplicates = []
        
        vendor_name = invoice_data.get('vendor_name', '')
        total_amount = float(invoice_data.get('total_amount', 0))
        invoice_date_str = invoice_data.get('invoice_date', '')
        
        if not vendor_name or not total_amount:
            return duplicates
        
        # Parse invoice date
        try:
            invoice_date = datetime.strptime(invoice_date_str, '%Y-%m-%d')
        except (ValueError, TypeError):
            # If date parsing fails, use today's date
            invoice_date = datetime.now()
        
        # Calculate amount range (±5%)
        amount_min = total_amount * (1 - AMOUNT_TOLERANCE)
        amount_max = total_amount * (1 + AMOUNT_TOLERANCE)
        
        # Calculate date threshold
        date_threshold = invoice_date - timedelta(days=TIME_WINDOW_DAYS)
        
        # Find fuzzy matches
        matches = self.historical_df[
            (self.historical_df['vendor_name'].astype(str) == str(vendor_name)) &
            (self.historical_df['total_amount'].astype(float) >= amount_min) &
            (self.historical_df['total_amount'].astype(float) <= amount_max) &
            (self.historical_df['invoice_date'] >= date_threshold)
        ]
        
        for _, match in matches.iterrows():
            match_amount = float(match['total_amount'])
            amount_diff_pct = abs(match_amount - total_amount) / total_amount
            
            # Confidence decreases with amount difference
            confidence = 1.0 - (amount_diff_pct / AMOUNT_TOLERANCE)
            
            duplicates.append(DuplicateError(
                strategy='FUZZY_AMOUNT_MATCH',
                confidence=confidence,
                match_data={
                    'similarity_score': confidence,
                    'amount_difference': abs(match_amount - total_amount),
                    'amount_diff_percent': amount_diff_pct * 100
                },
                original_invoice=match.to_dict()
            ))
            logger.warning(
                f"⚠️ FUZZY MATCH: Similar invoice from {vendor_name} "
                f"(${match_amount} vs ${total_amount}, diff: {amount_diff_pct*100:.1f}%)"
            )
        
        return duplicates
    
    def _detect_similar_invoice_number(self, invoice_data: Dict[str, Any]) -> List[DuplicateError]:
        """
        Strategy 3: Similar invoice numbers (typo detection)
        
        This catches duplicates where OCR misread the invoice number
        """
        duplicates = []
        
        invoice_number = invoice_data.get('invoice_number', '')
        vendor_name = invoice_data.get('vendor_name', '')
        
        if not invoice_number or not vendor_name or len(invoice_number) < 3:
            return duplicates
        
        # Find invoices from same vendor
        vendor_invoices = self.historical_df[
            self.historical_df['vendor_name'].astype(str) == str(vendor_name)
        ]
        
        for _, match in vendor_invoices.iterrows():
            match_invoice_number = str(match['invoice_number'])
            
            # Calculate similarity using Levenshtein distance
            similarity = fuzz.ratio(invoice_number, match_invoice_number) / 100.0
            
            # If very similar (>90%) but not exact match
            if similarity > 0.90 and invoice_number != match_invoice_number:
                duplicates.append(DuplicateError(
                    strategy='SIMILAR_INVOICE_NUMBER',
                    confidence=similarity,
                    match_data={
                        'similarity_score': similarity,
                        'matched_invoice_number': match_invoice_number
                    },
                    original_invoice=match.to_dict()
                ))
                logger.warning(
                    f"⚠️ TYPO DETECTED: Invoice numbers very similar "
                    f"('{invoice_number}' vs '{match_invoice_number}', "
                    f"similarity: {similarity:.2%})"
                )
        
        return duplicates


def detect_duplicates(invoice_data: Dict[str, Any], 
                     historical_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function for duplicate detection
    
    Args:
        invoice_data: Invoice to check for duplicates
        historical_path: Path to historical invoices CSV
        
    Returns:
        Detection results
    """
    detector = DuplicateDetector(historical_path)
    return detector.detect_duplicates(invoice_data)


# CLI support for testing
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python duplicate_detection.py <invoice_json_file> [historical_csv]")
        sys.exit(1)
    
    invoice_file = Path(sys.argv[1])
    historical_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Load invoice
    with open(invoice_file, 'r') as f:
        invoice = json.load(f)
    
    # Detect duplicates
    result = detect_duplicates(invoice, historical_file)
    
    print("\n" + "="*60)
    print("DUPLICATE DETECTION REPORT")
    print("="*60)
    print(f"Status: {'🚨 DUPLICATE FOUND' if result['is_duplicate'] else '✅ NO DUPLICATES'}")
    print(f"Duplicates Found: {result['duplicate_count']}")
    print(f"Highest Confidence: {result['highest_confidence']:.2%}")
    
    if result['is_duplicate']:
        print("\nDuplicate Details:")
        for i, dup in enumerate(result['duplicates_found'], 1):
            print(f"\n  {i}. {dup['strategy']}")
            print(f"     Confidence: {dup['confidence']:.2%}")
            print(f"     Original Invoice: {dup['match_details']['original_invoice_number']}")
            print(f"     Original Amount: ${dup['match_details']['original_amount']}")
    
    print("\n" + "="*60)