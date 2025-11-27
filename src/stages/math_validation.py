# src/stages/math_validation.py

"""
LedgerX Math Validation Module
Validates mathematical correctness of invoice calculations
Enhanced with per-field confidence scoring
"""

import logging
from typing import Dict, List, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Tolerance for floating-point comparisons (1 cent)
TOLERANCE = 0.01


class MathValidationError:
    """Represents a single math validation error"""
    
    def __init__(self, error_type: str, expected: float, actual: float, field: str):
        self.error_type = error_type
        self.expected = expected
        self.actual = actual
        self.field = field
        self.difference = abs(expected - actual)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.error_type,
            'field': self.field,
            'expected': round(self.expected, 2),
            'actual': round(self.actual, 2),
            'difference': round(self.difference, 2)
        }


class InvoiceMathValidator:
    """Validates invoice mathematical calculations"""
    
    def __init__(self, tolerance: float = TOLERANCE):
        self.tolerance = tolerance
        self.errors: List[MathValidationError] = []
    
    def validate(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main validation method
        
        Args:
            invoice_data: Dictionary containing invoice fields
            
        Returns:
            Validation result dictionary
        """
        self.errors = []
        
        # Extract required fields
        try:
            line_items = invoice_data.get('line_items', [])
            subtotal = float(invoice_data.get('subtotal', 0))
            tax_amount = float(invoice_data.get('tax_amount', 0))
            total_amount = float(invoice_data.get('total_amount', 0))
            tax_rate = float(invoice_data.get('tax_rate', 0))
            discount_amount = float(invoice_data.get('discount_amount', 0))
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse invoice amounts: {e}")
            return {
                'is_valid': False,
                'errors': [{'type': 'PARSING_ERROR', 'message': str(e)}],
                'confidence': 0.0
            }
        
        # Validation checks
        self._validate_line_items_sum(line_items, subtotal)
        self._validate_tax_calculation(subtotal, tax_rate, tax_amount, discount_amount)
        self._validate_total(subtotal, tax_amount, total_amount, discount_amount)
        
        # Calculate confidence score
        confidence = self._calculate_confidence()
        
        return {
            'is_valid': len(self.errors) == 0,
            'errors': [error.to_dict() for error in self.errors],
            'error_count': len(self.errors),
            'confidence': confidence,
            'validation_details': {
                'line_items_validated': len(line_items),
                'total_checks_performed': 3,
                'checks_passed': 3 - len(self.errors)
            }
        }
    
    def _validate_line_items_sum(self, line_items: List[Dict], subtotal: float):
        """Check if line items sum equals subtotal"""
        if not line_items:
            logger.warning("No line items found in invoice")
            return
        
        try:
            calculated_sum = sum([
                float(item.get('amount', 0)) 
                for item in line_items
            ])
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to calculate line items sum: {e}")
            return
        
        if abs(calculated_sum - subtotal) > self.tolerance:
            error = MathValidationError(
                error_type='LINE_ITEMS_MISMATCH',
                expected=calculated_sum,
                actual=subtotal,
                field='subtotal'
            )
            self.errors.append(error)
            logger.warning(f"Line items sum mismatch: {error.to_dict()}")
    
    def _validate_tax_calculation(self, subtotal: float, tax_rate: float, 
                                   tax_amount: float, discount_amount: float):
        """Check if tax calculation is correct"""
        # Tax is calculated on (subtotal - discount)
        taxable_amount = subtotal - discount_amount
        
        if tax_rate > 0:
            expected_tax = taxable_amount * (tax_rate / 100)
            
            if abs(expected_tax - tax_amount) > self.tolerance:
                error = MathValidationError(
                    error_type='TAX_CALCULATION_MISMATCH',
                    expected=expected_tax,
                    actual=tax_amount,
                    field='tax_amount'
                )
                self.errors.append(error)
                logger.warning(f"Tax calculation mismatch: {error.to_dict()}")
    
    def _validate_total(self, subtotal: float, tax_amount: float, 
                        total_amount: float, discount_amount: float):
        """Check if total = subtotal - discount + tax"""
        expected_total = subtotal - discount_amount + tax_amount
        
        if abs(expected_total - total_amount) > self.tolerance:
            error = MathValidationError(
                error_type='TOTAL_CALCULATION_MISMATCH',
                expected=expected_total,
                actual=total_amount,
                field='total_amount'
            )
            self.errors.append(error)
            logger.warning(f"Total calculation mismatch: {error.to_dict()}")
    
    def _calculate_confidence(self) -> float:
        """Calculate validation confidence score"""
        if len(self.errors) == 0:
            return 1.0
        elif len(self.errors) == 1:
            # Single error might be rounding issue
            return 0.7
        else:
            # Multiple errors indicate serious problems
            return 0.3


def validate_invoice_math(invoice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for single invoice validation
    
    Args:
        invoice_data: Dictionary containing invoice fields
        
    Returns:
        Validation result
    """
    validator = InvoiceMathValidator()
    return validator.validate(invoice_data)


def validate_with_confidence(invoice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced validation that includes field-level confidence scoring
    
    Args:
        invoice_data: Invoice data with optional confidence scores
        
    Returns:
        Validation result with field confidence analysis
    """
    # Run standard math validation
    validator = InvoiceMathValidator()
    math_result = validator.validate(invoice_data)
    
    # Extract field-level confidence scores if available
    field_confidences = {}
    low_confidence_fields = []
    
    # Check for confidence in critical fields
    critical_fields = [
        'invoice_number',
        'vendor_name', 
        'total_amount',
        'tax_amount',
        'subtotal',
        'invoice_date'
    ]
    
    for field in critical_fields:
        # Check if field has confidence score
        confidence_key = f"{field}_confidence"
        if confidence_key in invoice_data:
            confidence = float(invoice_data[confidence_key])
            field_confidences[field] = confidence
            
            # Flag fields with low confidence (<70%)
            if confidence < 0.70:
                low_confidence_fields.append({
                    'field': field,
                    'confidence': confidence,
                    'value': str(invoice_data.get(field, 'N/A')),
                    'requires_review': True
                })
        elif field in invoice_data:
            # No confidence score available, assume from OCR confidence
            ocr_confidence = invoice_data.get('ocr_confidence', 0.85)
            field_confidences[field] = ocr_confidence
    
    # Calculate average confidence
    avg_confidence = (
        sum(field_confidences.values()) / len(field_confidences) 
        if field_confidences else 0.0
    )
    
    # Add field confidence analysis to result
    math_result['field_confidence_analysis'] = {
        'field_confidences': field_confidences,
        'low_confidence_fields': low_confidence_fields,
        'low_confidence_count': len(low_confidence_fields),
        'average_confidence': avg_confidence,
        'fields_requiring_review': [f['field'] for f in low_confidence_fields]
    }
    
    # Adjust overall confidence based on field confidence
    if low_confidence_fields:
        # Lower overall confidence if critical fields have low confidence
        field_confidence_penalty = 0.2 * len(low_confidence_fields) / len(critical_fields)
        math_result['confidence'] = max(0.0, math_result['confidence'] - field_confidence_penalty)
    
    return math_result


def validate_from_json_file(json_path: Path, use_confidence: bool = False) -> Dict[str, Any]:
    """
    Validate invoice from JSON file
    
    Args:
        json_path: Path to invoice JSON file
        use_confidence: Whether to use enhanced confidence validation
        
    Returns:
        Validation result
    """
    try:
        with open(json_path, 'r') as f:
            invoice_data = json.load(f)
        
        if use_confidence:
            return validate_with_confidence(invoice_data)
        else:
            return validate_invoice_math(invoice_data)
    
    except FileNotFoundError:
        logger.error(f"Invoice file not found: {json_path}")
        return {
            'is_valid': False,
            'errors': [{'type': 'FILE_NOT_FOUND', 'message': str(json_path)}],
            'confidence': 0.0
        }
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file: {json_path}")
        return {
            'is_valid': False,
            'errors': [{'type': 'INVALID_JSON', 'message': str(e)}],
            'confidence': 0.0
        }


# CLI support for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python math_validation.py <invoice_json_file> [--with-confidence]")
        print("\nOptions:")
        print("  --with-confidence    Enable per-field confidence analysis")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    use_confidence = '--with-confidence' in sys.argv
    
    # Load invoice
    with open(json_file, 'r') as f:
        invoice_data = json.load(f)
    
    # Run validation
    if use_confidence:
        result = validate_with_confidence(invoice_data)
    else:
        result = validate_invoice_math(invoice_data)
    
    # Display results
    print("\n" + "="*70)
    if use_confidence:
        print("ENHANCED MATH VALIDATION REPORT (WITH FIELD CONFIDENCE)")
    else:
        print("INVOICE MATH VALIDATION REPORT")
    print("="*70)
    print(f"Status: {'✅ VALID' if result['is_valid'] else '❌ INVALID'}")
    print(f"Overall Confidence: {result['confidence']:.2%}")
    print(f"Errors Found: {result.get('error_count', 0)}")
    
    # Field confidence analysis (if enabled)
    if 'field_confidence_analysis' in result:
        analysis = result['field_confidence_analysis']
        print("\n" + "-"*70)
        print("FIELD CONFIDENCE ANALYSIS")
        print("-"*70)
        print(f"Average Field Confidence: {analysis['average_confidence']:.2%}")
        print(f"Low Confidence Fields: {analysis['low_confidence_count']}")
        
        if analysis['low_confidence_fields']:
            print("\n⚠️  FIELDS REQUIRING HUMAN REVIEW:")
            for field in analysis['low_confidence_fields']:
                print(f"  • {field['field']:<20} {field['confidence']:>6.2%}  (value: {field['value']})")
        
        print("\n📊 ALL FIELD CONFIDENCES:")
        for field, conf in sorted(analysis['field_confidences'].items()):
            emoji = "✅" if conf >= 0.70 else "⚠️ "
            status = "OK" if conf >= 0.70 else "LOW"
            print(f"  {emoji} {field:<20} {conf:>6.2%}  [{status}]")
    
    # Calculation errors
    if not result['is_valid']:
        print("\n" + "-"*70)
        print("❌ CALCULATION ERRORS DETECTED")
        print("-"*70)
        for i, error in enumerate(result['errors'], 1):
            if isinstance(error, dict) and 'type' in error:
                print(f"\n  {i}. {error['type']}")
                if 'field' in error:
                    print(f"     Field: {error.get('field', 'N/A')}")
                if 'expected' in error and 'actual' in error:
                    print(f"     Expected: ${error.get('expected', 0):.2f}")
                    print(f"     Actual: ${error.get('actual', 0):.2f}")
                    print(f"     Difference: ${error.get('difference', 0):.2f}")
            else:
                print(f"\n  {i}. {error.get('message', str(error))}")
    
    # Validation details
    if 'validation_details' in result:
        details = result['validation_details']
        print("\n" + "-"*70)
        print("VALIDATION DETAILS")
        print("-"*70)
        print(f"  Line Items Validated: {details.get('line_items_validated', 0)}")
        print(f"  Total Checks Performed: {details.get('total_checks_performed', 0)}")
        print(f"  Checks Passed: {details.get('checks_passed', 0)}")
    
    print("\n" + "="*70)
    
    # Exit code
    sys.exit(0 if result['is_valid'] else 1)