import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import json
from datasets import load_dataset

class EnhancedMathRAGSystem:
    def __init__(self, processed_dataset):
        self.dataset = processed_dataset
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.embeddings = None
        
    def create_vector_index(self):
        """Create FAISS vector index for efficient similarity search"""
        texts = []
        for item in self.dataset:
            # Focus more on the question for better matching
            combined_text = f"Question: {item['question']}"
            texts.append(combined_text)
        
        print("Generating embeddings...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Index created with {self.index.ntotal} vectors")
    
    def retrieve_similar_problems(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve k most similar problems for a given query"""
        if self.index is None:
            raise ValueError("Vector index not created. Call create_vector_index() first.")
        
        query_embedding = self.model.encode([f"Question: {query}"])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            idx = int(idx)
            results.append({
                'rank': i + 1,
                'score': float(score),
                'question': self.dataset[idx]['question'],
                'answer': self.dataset[idx]['answer'],
                'context': self.dataset[idx].get('context', '')
            })
        
        return results
    
    def extract_solution_pattern(self, answer: str) -> str:
        """Extract the general solution pattern from an answer"""
        # Look for mathematical operations and patterns
        patterns = []
        
        # Common mathematical operations
        if re.search(r'\d+\s*\*\s*\d+', answer):
            patterns.append("multiplication")
        if re.search(r'\d+\s*\+\s*\d+', answer):
            patterns.append("addition")
        if re.search(r'\d+\s*-\s*\d+', answer):
            patterns.append("subtraction")
        if re.search(r'\d+\s*/\s*\d+', answer):
            patterns.append("division")
        if re.search(r'%|percent', answer.lower()):
            patterns.append("percentage")
        
        return ", ".join(patterns) if patterns else "general arithmetic"
    
    def solve_simple_problems(self, query: str) -> str:
        """Attempt to solve simple mathematical problems directly"""
        query_lower = query.lower()
        
        # Extract numbers from the query
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        numbers = [float(n) for n in numbers]
        
        # Simple multiplication problems (like cookies × price)
        if ('sold' in query_lower or 'sell' in query_lower) and len(numbers) >= 2:
            if 'each' in query_lower or '@' in query_lower or 'per' in query_lower:
                quantity, price = numbers[0], numbers[1]
                total = quantity * price
                return f"""**Direct Solution:**
You sold {int(quantity)} items at ${price} each.
Total earned = {int(quantity)} * ${price} = ${total}

**Answer: ${total}**"""
        
        # Percentage problems
        if '%' in query or 'percent' in query_lower:
            if len(numbers) >= 2:
                total, percentage = numbers[0], numbers[1]
                if 'gives away' in query_lower or 'give away' in query_lower:
                    given_away = total * (percentage / 100)
                    remaining = total - given_away
                    return f"""**Direct Solution:**
Starting amount: {int(total)}
Percentage given away: {percentage}%
Amount given away = {int(total)} * {percentage}% = {int(total)} × {percentage/100} = {given_away}
Amount remaining = {int(total)} - {given_away} = {remaining}

**Answer: {int(remaining)} items left**"""
        
        # Ratio/proportion problems (like recipe scaling)
        if 'recipe' in query_lower or 'cups' in query_lower:
            if len(numbers) >= 3:
                ingredient_amount, original_quantity, new_quantity = numbers[0], numbers[1], numbers[2]
                ratio = ingredient_amount / original_quantity
                new_amount = ratio * new_quantity
                return f"""**Direct Solution:**
Original recipe: {ingredient_amount} cups for {int(original_quantity)} items
Ratio: {ingredient_amount}/{int(original_quantity)} = {ratio} cups per item
For {int(new_quantity)} items: {ratio} * {int(new_quantity)} = {new_amount} cups

**Answer: {new_amount} cups needed**"""
        
        # Compound interest
        if 'compound interest' in query_lower:
            if len(numbers) >= 3:
                principal, rate, years = numbers[0], numbers[1], numbers[2]
                # A = P(1 + r/100)^t
                amount = principal * ((1 + rate/100) ** years)
                interest = amount - principal
                return f"""**Direct Solution:**
Principal (P) = ${principal}
Rate (r) = {rate}% per year  
Time (t) = {years} years
Compound Interest Formula: A = P(1 + r/100)^t
A = ${principal} * (1 + {rate}/100)^{years}
A = ${principal} * (1.{int(rate):02d})^{years}
A = ${principal} * {(1 + rate/100)**years:.4f}
A = ${amount:.2f}
Compound Interest = ${amount:.2f} - ${principal} = ${interest:.2f}

**Answer: ${interest:.2f} compound interest**"""
        
        return None
    
    def generate_enhanced_response(self, query: str, k: int = 3) -> str:
        """Generate enhanced response with direct solution when possible"""
        
        # Try to solve directly first
        direct_solution = self.solve_simple_problems(query)
        
        # Get similar problems for context
        similar_problems = self.retrieve_similar_problems(query, k)
        
        response = f"## Query: {query}\n\n"
        
        if direct_solution:
            response += direct_solution + "\n\n"
            response += "---\n\n"
        
        response += "## Similar Problems for Reference:\n\n"
        
        for i, problem in enumerate(similar_problems):
            solution_pattern = self.extract_solution_pattern(problem['answer'])
            response += f"**Example {i+1}** (Similarity: {problem['score']:.3f}, Pattern: {solution_pattern}):\n"
            response += f"*Question:* {problem['question']}\n"
            response += f"*Solution:* {problem['answer']}\n\n"
        
        if not direct_solution:
            response += f"\n**Suggested Approach:**\nBased on the most similar problem (similarity: {similar_problems[0]['score']:.3f}), "
            response += "you can follow a similar step-by-step approach to solve your specific question.\n"
        
        return response
    
    def analyze_query_types(self, queries: List[str]) -> Dict:
        """Analyze what types of problems the system can handle"""
        analysis = {
            'direct_solvable': [],
            'needs_context': [],
            'problem_types': {}
        }
        
        for query in queries:
            direct_solution = self.solve_simple_problems(query)
            if direct_solution:
                analysis['direct_solvable'].append(query)
            else:
                analysis['needs_context'].append(query)
            
            # Classify problem type
            query_lower = query.lower()
            if 'sold' in query_lower or 'sell' in query_lower:
                problem_type = 'sales_revenue'
            elif '%' in query or 'percent' in query_lower:
                problem_type = 'percentage'
            elif 'interest' in query_lower:
                problem_type = 'interest_calculation'
            elif 'recipe' in query_lower or 'cups' in query_lower:
                problem_type = 'ratio_proportion'
            else:
                problem_type = 'general_math'
            
            analysis['problem_types'][problem_type] = analysis['problem_types'].get(problem_type, 0) + 1
        
        return analysis

def test_enhanced_system(processed_dataset):
    """Test the enhanced system"""
    print("=== Enhanced RAG System Test ===\n")
    
    # Initialize enhanced system
    enhanced_rag = EnhancedMathRAGSystem(processed_dataset)
    enhanced_rag.create_vector_index()
    
    # Test queries
    test_queries = [
        "How much money did I make if I sold 100 cookies at $2 each?",
        "A person has 50 apples and gives away 20% of them. How many are left?",
        "Calculate compound interest on $1000 at 5% for 2 years",
        "If a recipe needs 3 cups of flour for 12 cookies, how much for 20 cookies?"
    ]
    
    # Analyze query types
    analysis = enhanced_rag.analyze_query_types(test_queries)
    print("Query Analysis:")
    print(f"Directly solvable: {len(analysis['direct_solvable'])}")
    print(f"Need context: {len(analysis['needs_context'])}")
    print(f"Problem types: {analysis['problem_types']}")
    print("\n" + "="*80 + "\n")
    
    # Generate enhanced responses
    for query in test_queries:
        response = enhanced_rag.generate_enhanced_response(query)
        print(response)
        print("="*80 + "\n")

def compare_systems_performance(processed_dataset, test_queries):
    """Compare original vs enhanced system performance"""
    print("=== System Performance Comparison ===\n")
    
    # This would require running both systems and comparing:
    # 1. Response relevance
    # 2. Direct problem solving capability
    # 3. User satisfaction (qualitative)
    
    metrics = {
        'original_system': {
            'retrieval_accuracy': 'High (good semantic matching)',
            'direct_solving': 'None (only retrieves examples)',
            'response_format': 'Context-heavy, requires interpretation'
        },
        'enhanced_system': {
            'retrieval_accuracy': 'High (focused on questions)',
            'direct_solving': 'Yes (for common problem types)',
            'response_format': 'Direct solution + context'
        }
    }
    
    for system, metrics_dict in metrics.items():
        print(f"## {system.replace('_', ' ').title()}:")
        for metric, value in metrics_dict.items():
            print(f"- {metric.replace('_', ' ').title()}: {value}")
        print()

def split_prompt(example):
    text = example['clear_prompt']
    
    # Extract sections using regex with multiline and dotall
    question_match = re.search(r'##QUESTION##\s*(.*?)\s*##ANSWER##', text, re.DOTALL)
    answer_match = re.search(r'##ANSWER##\s*(.*)', text, re.DOTALL)
    context_match = re.search(r'##CONTEXT##\s*(.*?)\s*##QUESTION##', text, re.DOTALL)

    return {
        'context': context_match.group(1).strip() if context_match else "",
        'question': question_match.group(1).strip() if question_match else "",
        'answer': answer_match.group(1).strip() if answer_match else ""
    }

# Usage
if __name__ == "__main__":
    dataset = load_dataset("neural-bridge/rag-full-20000", split="train")
    processed_dataset = dataset.map(split_prompt)
    # Assuming you have processed_dataset available
    test_enhanced_system(processed_dataset)
    pass
