try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from economic_hmm_analysis import EconomicHMMAnalyzer
    print("Import successful!")
except SyntaxError as e:
    print(f"Syntax error: {e}")
except Exception as e:
    print(f"Other error: {e}")
