#!/usr/bin/env python3
"""
Simple test script to verify the Thought Anchors setup.
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from config.models import ModelConfig
        from config.experiments import ExperimentConfig
        print("✓ Config modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import config modules: {e}")
        return False
    
    try:
        from src.sentence_analysis import SentenceAnalyzer
        print("✓ Sentence analysis module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sentence analysis: {e}")
        return False
    
    try:
        from src.blackbox import BlackboxResampler, ImportanceAnalyzer
        print("✓ Blackbox modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import blackbox modules: {e}")
        return False
    
    try:
        from src.attention import ReceiverHeadsAnalyzer, AttentionSuppressionAnalyzer
        print("✓ Attention modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import attention modules: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality with dummy data."""
    print("\nTesting basic functionality...")
    
    try:
        from src.sentence_analysis import SentenceAnalyzer
        
        # Test sentence analysis
        analyzer = SentenceAnalyzer(use_llm_labeling=False)
        test_text = "Let me solve this step by step. First, I need to understand the problem. Then I'll calculate the answer."
        
        labeled_sentences = analyzer.analyze_reasoning_trace(test_text)
        print(f"✓ Sentence analysis completed: {len(labeled_sentences)} sentences labeled")
        
        # Test importance calculation
        importance_scores = analyzer.get_sentence_importance(labeled_sentences)
        print(f"✓ Importance scores calculated: {len(importance_scores)} scores")
        
        # Test thought anchors identification
        thought_anchors = analyzer.get_thought_anchors(labeled_sentences)
        print(f"✓ Thought anchors identified: {len(thought_anchors)} anchors")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config.models import ModelConfig, DEFAULT_CONFIGS
        from config.experiments import ExperimentConfig, DEFAULT_EXPERIMENTS
        
        # Test default configs
        model_config = ModelConfig()
        print(f"✓ Model config created: {model_config.reasoning_model_name}")
        
        experiment_config = ExperimentConfig()
        print(f"✓ Experiment config created: {experiment_config.experiment_name}")
        
        # Test preset configs
        print(f"✓ Available model configs: {list(DEFAULT_CONFIGS.keys())}")
        print(f"✓ Available experiment configs: {list(DEFAULT_EXPERIMENTS.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Thought Anchors Setup Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1
            print(f"✓ {test_name} test passed")
        else:
            print(f"✗ {test_name} test failed")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is working correctly.")
        print("\nNext steps:")
        print("1. Copy env.example to .env and configure your settings")
        print("2. Install a language model or configure API access")
        print("3. Run experiments with: python run_experiments.py --reasoning_trace data/sample_reasoning_trace.txt")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 