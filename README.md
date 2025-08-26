# MyndraComposer 🍎

An AI-powered cross-platform tool that analyzes source code files and translates them into idiomatic [Myndra programming language](../Myndra2[c]/) code using machine learning techniques.

## Features

- **Multi-language Analysis**: Supports 20+ programming languages including Python, JavaScript, Java, C++, Rust, Go, and more
- **ML-Powered Translation**: Uses Ollama models (with extensible support for other providers) to understand code semantics and generate appropriate Myndra code
- **Interactive CLI**: Claude Code-style interactive assistant for guided code generation
- **Project State Management**: Persistent `.pomuse` folders for continuing work across sessions
- **Cross-Platform**: Works on Windows, Linux, macOS, and BSD systems
- **Domain Detection**: Automatically identifies code domains (web dev, math/scientific, data processing, etc.)
- **Smart Templates**: Context-aware code generation using appropriate Myndra patterns

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is running** (default setup):
   ```bash
   # Install Ollama if not already installed
   # Then start with a code-understanding model
   ollama pull codellama
   ollama serve
   ```

3. **Start MyndraComposer**:
   ```bash
   python pomuse.py
   ```

4. **Analyze your code**:
   ```
   pomuse> analyze ./src
   ```

5. **Generate Myndra code**:
   ```
   pomuse> generate "create a robust math framework from these files"
   ```

## Commands

- `analyze [path]` - Analyze source code files in the specified directory
- `generate "<prompt>"` - Generate Myndra code based on analysis and your prompt
- `continue` - Resume work from a previous session (uses `.pomuse` state)
- `status` - Show current project status and configuration
- `help` - Show available commands
- `exit` - Exit the application

## Project Structure

When you run MyndraComposer in a directory, it creates a `.pomuse` folder containing:

```
.pomuse/
├── project_state.json     # Project configuration and history
├── conversations/         # Chat history and context
└── outputs/              # Generated Myndra files
```

## Example Workflow

```bash
# 1. Navigate to your project directory
cd /path/to/your/code

# 2. Start MyndraComposer
python /path/to/pomuse.py

# 3. Analyze existing code
pomuse> analyze .

# 4. Generate Myndra equivalent
pomuse> generate "these files implement a web API - create a Myndra version with reactive UI and capability-based security"

# 5. Continue working later
pomuse> continue
```

## Supported Languages

- **Web**: JavaScript, TypeScript, HTML, CSS
- **Systems**: C, C++, Rust, Go
- **Enterprise**: Java, C#, Scala, Kotlin
- **Scripting**: Python, Ruby, PHP, Bash
- **Functional**: Haskell, OCaml, F#, Clojure
- **Scientific**: R, Julia, MATLAB
- **Data**: SQL, JSON, XML, YAML
- **Other**: Swift, Lua, Perl

## Configuration

MyndraComposer stores configuration in `.pomuse/project_state.json`:

```json
{
  "settings": {
    "model_provider": "ollama",
    "default_model": "codellama", 
    "auto_execute_builds": false,
    "remember_file_permissions": true
  }
}
```

## Integration with Myndra

Generated code uses modern Myndra features:

- **Reactive Programming**: `@reactive` functions and observables
- **Temporal Types**: `evolving<T>` for animations and state transitions  
- **Capability-Based Security**: Fine-grained permissions with `capabilities(...)`
- **Context-Aware Code**: Automatic dev/prod/test environment adaptation
- **Live Code Capsules**: Hot-reloadable UI components
- **Semantic Tags**: `#tag:` annotations for better organization
- **Fallback Strategies**: Built-in error recovery with `fallback` clauses
- **DSL Support**: Inline SQL, HTML, CSS, and shader code blocks

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File System   │    │   ML Provider    │    │   Myndra        │
│                 │    │   (Ollama)       │    │   Generator     │
│ ┌─────────────┐ │    │                  │    │                 │
│ │Source Files │ │──▶ │ ┌──────────────┐ │──▶ │ ┌─────────────┐ │
│ └─────────────┘ │    │ │Code Analysis │ │    │ │Generated    │ │
│                 │    │ │& Translation │ │    │ │.myn Files   │ │
└─────────────────┘    │ └──────────────┘ │    │ └─────────────┘ │
                       └──────────────────┘    └─────────────────┘
                                ▲
                                │
                       ┌──────────────────┐
                       │ Interactive CLI  │
                       │ & State Manager  │
                       └──────────────────┘
```

## Future Enhancements

- **Multiple ML Providers**: Support for OpenAI, Anthropic, Google, etc.
- **Build Integration**: Automatic testing and building of generated code
- **Code Refinement**: Iterative improvement based on build feedback
- **Template System**: User-defined code generation templates
- **Plugin Architecture**: Extensible analysis and generation plugins
- **Web Interface**: Optional GUI for visual code exploration
- **Version Control**: Git integration for tracking generated code evolution

## Contributing

MyndraComposer is designed to be extensible. Key areas for contribution:

1. **Language Support**: Add new source language analyzers
2. **ML Providers**: Integrate additional AI model providers  
3. **Templates**: Create domain-specific Myndra code templates
4. **Analysis**: Improve semantic code understanding
5. **Testing**: Add comprehensive test coverage

## License

This project is part of the Myndra ecosystem. See the main Myndra repository for licensing information.