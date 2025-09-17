#!/usr/bin/env python3
"""
Query the Finance QA API with a list of questions and collect answers.
"""

import requests
import json
import time
import sys

# API configuration
API_URL = 'http://localhost:8000'

# Questions to ask
questions = [
    "What is the total revenue of Berkshire in 2021",
    'What is the largest risk to Google?',
    "What, in percentage, is Google's reliance on advertising",
    "Who is Bertie?",
    "List Berkshire's Securities registered pursuant to Section 12(b) and Section 12(g) of the Act of 1934",
    "Compare Alphabet's revenue to American Express"
]

def ask_question(question: str) -> dict:
    """Ask a question via the async API and wait for the answer."""
    print(f"\n🤔 Question: {question}")
    print("-" * 60)
    
    try:
        # Submit question asynchronously
        response = requests.post(f"{API_URL}/question_async", json={'question': question})
        if not response.ok:
            print(f"❌ Error submitting question: {response.text}")
            return {"error": f"API error: {response.text}"}
        
        job_id = response.json()['job_id']
        print(f"📝 Job ID: {job_id}")
        
        # Poll for completion
        max_polls = 120  # 2 minutes max
        poll_interval = 1.0
        
        for poll in range(max_polls):
            time.sleep(poll_interval)
            
            # Check job status
            job_response = requests.get(f"{API_URL}/jobs/{job_id}")
            if not job_response.ok:
                print(f"❌ Error checking job status: {job_response.text}")
                continue
            
            events = job_response.json().get('events', [])
            
            # Check if job is finished
            trace_id = None
            finished = False
            
            for event in events:
                if event['event'] == 'job_finished':
                    finished = True
                elif event['event'] == 'info' and 'trace_id' in event.get('data', {}):
                    trace_id = event['data']['trace_id']
            
            if finished and trace_id:
                print(f"✅ Job completed. Trace ID: {trace_id}")
                
                # Get the full trace with answer
                trace_response = requests.get(f"{API_URL}/trace/{trace_id}")
                if trace_response.ok:
                    trace_data = trace_response.json()
                    final_answer = trace_data.get('final_answer', {})
                    
                    if isinstance(final_answer, dict):
                        answer = final_answer.get('answer', 'No answer provided')
                        reasoning = final_answer.get('reasoning', 'No reasoning provided')
                    else:
                        answer = str(final_answer) if final_answer else 'No answer provided'
                        reasoning = 'No reasoning provided'
                    
                    print(f"💡 Answer: {answer}")
                    if reasoning and reasoning != 'No reasoning provided':
                        print(f"🧠 Reasoning: {reasoning}")
                    
                    return {
                        "question": question,
                        "answer": answer,
                        "reasoning": reasoning,
                        "trace_id": trace_id,
                        "job_id": job_id
                    }
                else:
                    print(f"❌ Error fetching trace: {trace_response.text}")
                    return {"error": f"Error fetching trace: {trace_response.text}"}
            
            # Show progress
            if poll % 5 == 0:
                print(f"⏳ Waiting... ({poll}/{max_polls})")
        
        print(f"⏰ Timeout waiting for answer")
        return {"error": "Timeout waiting for answer"}
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error. Is the server running at {API_URL}?")
        return {"error": "Connection error - server not running"}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}"}

def main():
    print("🚀 Finance QA System - Batch Question Processing")
    print("=" * 60)
    
    # Check if server is running
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.ok:
            print(f"✅ Server is running at {API_URL}")
        else:
            print(f"⚠️  Server responded but may have issues: {health_response.status_code}")
    except:
        print(f"❌ Cannot connect to server at {API_URL}")
        print("Please start the server with: uvicorn app.api.server:app --port 8000")
        return
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(questions)}")
        result = ask_question(question)
        results.append(result)
        
        # Brief pause between questions
        if i < len(questions):
            time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY OF ALL ANSWERS")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.get('question', 'Unknown question')}")
        if 'error' in result:
            print(f"   ❌ {result['error']}")
        else:
            print(f"   💡 {result.get('answer', 'No answer')}")
    
    # Save results to file
    with open('qa_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to qa_results.json")

if __name__ == '__main__':
    main()