#!/usr/bin/env python3
"""
Demo script for MyndraComposer
Shows the complete workflow from code analysis to Myndra generation
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from pomuse import InteractiveCLI, CodeAnalyzer, EnhancedOllamaProvider, MyndraGenerator


async def run_automated_demo():
    """Run an automated demo of MyndraComposer capabilities"""
    print("🍎 MyndraComposer Automated Demo")
    print("=" * 50)
    
    # Initialize components
    working_dir = Path(__file__).parent
    analyzer = CodeAnalyzer()
    ollama = EnhancedOllamaProvider()
    generator = MyndraGenerator(ollama)
    
    # Initialize Ollama
    print("\n1. Initializing ML provider...")
    try:
        if await ollama.initialize():
            print(f"✅ Connected to Ollama (model: {ollama.default_model})")
        else:
            print("⚠️  Ollama not available - using fallback templates")
    except Exception as e:
        print(f"⚠️  Ollama error: {e}")
    
    # Analyze test file
    print("\n2. Analyzing sample code...")
    test_file = working_dir / "test_sample.py"
    
    if test_file.exists():
        analysis = await analyzer.analyze_file(test_file)
        print(f"✅ Analyzed {test_file.name}: {analysis.get('language', 'unknown')} file")
        print(f"   Lines: {analysis.get('line_count', 0)}")
        print(f"   Domains: {analysis.get('domain_hints', [])}")
        
        # Get ML analysis if available
        if not analysis.get('error'):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                ml_analysis = await ollama.analyze_file_with_ml(
                    str(test_file), content, analysis.get('language', 'python')
                )
                print(f"   ML Purpose: {ml_analysis.get('primary_purpose', 'Unknown')}")
                print(f"   ML Complexity: {ml_analysis.get('complexity_level', 'Unknown')}")
                
                # Store enhanced analysis
                enhanced_analysis = {**analysis, 'ml_analysis': ml_analysis}
                
            except Exception as e:
                print(f"   ML analysis failed: {e}")
                enhanced_analysis = analysis
        else:
            enhanced_analysis = analysis
    else:
        print(f"❌ Test file not found: {test_file}")
        return
    
    # Generate context analysis
    print("\n3. Generating project context...")
    user_prompt = "Create a robust math framework with reactive UI from this API code"
    
    try:
        context_analysis = await ollama.analyze_code_context([enhanced_analysis], user_prompt)
        print(f"✅ Project type: {context_analysis.get('project_type', 'Unknown')}")
        
        template_type = context_analysis.get('translation_strategy', {}).get('template_type', 'basic_app')
        print(f"   Recommended template: {template_type}")
        
        features = context_analysis.get('translation_strategy', {}).get('myndra_features_to_use', [])
        print(f"   Suggested features: {', '.join(features) if features else 'basic'}")
        
    except Exception as e:
        print(f"⚠️  Context analysis failed: {e}")
        context_analysis = {"translation_strategy": {"template_type": "basic_app"}}
    
    # Generate Myndra code
    print("\n4. Generating Myndra code...")
    
    try:
        full_context = {
            'files': [enhanced_analysis],
            'ml_analysis': context_analysis
        }
        
        generated_code = await generator.generate_code(full_context, user_prompt)
        
        # Save generated code
        output_file = working_dir / "demo_output.myn"
        with open(output_file, 'w') as f:
            f.write(generated_code)
        
        print(f"✅ Generated Myndra code saved to: {output_file}")
        print("\n5. Generated code preview:")
        print("-" * 40)
        
        # Show first 30 lines
        lines = generated_code.split('\n')
        preview_lines = lines[:30]
        for i, line in enumerate(preview_lines, 1):
            print(f"{i:2d}│ {line}")
        
        if len(lines) > 30:
            print(f"   ... ({len(lines) - 30} more lines)")
        
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
    
    print("\n✅ Demo completed!")
    print(f"\nTo try the interactive mode, run: python pomuse.py")


async def run_interactive_demo():
    """Run interactive CLI demo"""
    print("🍎 Starting MyndraComposer Interactive Demo")
    print("=" * 50)
    
    cli = InteractiveCLI()
    working_dir = Path(__file__).parent
    
    print("Commands to try:")
    print("1. analyze .               # Analyze current directory")
    print("2. generate \"create a math framework\"  # Generate code")
    print("3. status                  # Check project status")
    print("4. help                    # Show all commands")
    print("5. exit                    # Exit")
    print()
    
    await cli.start_interactive_session(working_dir)


def main():
    """Main demo entry point"""
    print("Choose demo mode:")
    print("1. Automated demo (shows full workflow)")
    print("2. Interactive CLI demo")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            asyncio.run(run_automated_demo())
        elif choice == "2":
            asyncio.run(run_interactive_demo())
        else:
            print("Invalid choice. Defaulting to automated demo.")
            asyncio.run(run_automated_demo())
            
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    main()