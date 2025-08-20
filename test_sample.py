#!/usr/bin/env python3
"""
Sample Python code for testing PomegranteMuse translation capabilities.
This represents a simple web API with mathematical calculations.
"""

import math
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MathResult:
    """Result of a mathematical operation"""
    operation: str
    inputs: List[float]
    result: float
    timestamp: datetime


class MathCalculator:
    """Handles various mathematical calculations"""
    
    def __init__(self):
        self.history: List[MathResult] = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        result = a + b
        self.history.append(MathResult("add", [a, b], result, datetime.now()))
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        result = a * b
        self.history.append(MathResult("multiply", [a, b], result, datetime.now()))
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Calculate base^exponent"""
        result = math.pow(base, exponent)
        self.history.append(MathResult("power", [base, exponent], result, datetime.now()))
        return result
    
    def sqrt(self, value: float) -> float:
        """Calculate square root"""
        if value < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(value)
        self.history.append(MathResult("sqrt", [value], result, datetime.now()))
        return result
    
    def get_history(self) -> List[Dict]:
        """Get calculation history as JSON-serializable dict"""
        return [
            {
                "operation": item.operation,
                "inputs": item.inputs,
                "result": item.result,
                "timestamp": item.timestamp.isoformat()
            }
            for item in self.history
        ]
    
    def clear_history(self):
        """Clear calculation history"""
        self.history.clear()


class WebAPIHandler:
    """Simple web API handler for math operations"""
    
    def __init__(self):
        self.calculator = MathCalculator()
        self.request_count = 0
    
    def handle_request(self, operation: str, params: Dict) -> Dict:
        """Handle incoming API request"""
        self.request_count += 1
        
        try:
            if operation == "add":
                result = self.calculator.add(params["a"], params["b"])
            elif operation == "multiply":
                result = self.calculator.multiply(params["a"], params["b"])
            elif operation == "power":
                result = self.calculator.power(params["base"], params["exponent"])
            elif operation == "sqrt":
                result = self.calculator.sqrt(params["value"])
            elif operation == "history":
                result = self.calculator.get_history()
            elif operation == "stats":
                result = {
                    "total_requests": self.request_count,
                    "history_count": len(self.calculator.history)
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": True,
                "operation": operation,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": operation,
                "timestamp": datetime.now().isoformat()
            }
    
    def process_batch(self, requests: List[Dict]) -> List[Dict]:
        """Process multiple requests in batch"""
        results = []
        for req in requests:
            operation = req.get("operation")
            params = req.get("params", {})
            result = self.handle_request(operation, params)
            results.append(result)
        return results


def main():
    """Main application entry point"""
    print("Math API Server Starting...")
    
    api = WebAPIHandler()
    
    # Example usage
    test_requests = [
        {"operation": "add", "params": {"a": 5, "b": 3}},
        {"operation": "multiply", "params": {"a": 4, "b": 7}},
        {"operation": "power", "params": {"base": 2, "exponent": 8}},
        {"operation": "sqrt", "params": {"value": 16}},
        {"operation": "stats", "params": {}}
    ]
    
    print("Processing test requests...")
    results = api.process_batch(test_requests)
    
    for result in results:
        print(f"Result: {json.dumps(result, indent=2)}")
    
    print(f"\nTotal requests processed: {api.request_count}")
    print("Math API Server finished.")


if __name__ == "__main__":
    main()