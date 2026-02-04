import json
import os
import time
from pathlib import Path
from typing import Dict, List

import ollama


def extract_commands_from_file(file_path: Path) -> List[str]:
    """
    Extract individual command lines from a file.
    Each non-empty, non-comment line is treated as a separate command.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            print(f"    Error reading file: {e}")
            return []

    commands = []
    for line in content.split("\n"):
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith("#"):
            commands.append(line)

    return commands


def analyze_single_command(
    command: str,
    filename: str,
    command_num: int,
    total_commands: int,
    model: str = "gemma3:27b",
    retry_count: int = 3,
) -> Dict[str, str]:
    """
    Use Ollama to analyze a single command line.
    """
    # Improved prompt for better responses
    prompt = f"""You are a TagTool expert for Halo modding. Describe what this command does in one clear sentence. Do not use pronouns such as it and do not reference tagtool.

Command: {command}

Answer with ONLY the description, nothing else:"""

    print(
        f"\n    [{command_num}/{total_commands}] Command: {command[:80]}{'...' if len(command) > 80 else ''}"
    )

    for attempt in range(retry_count):
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 300,  # Increased to allow fuller responses
                    "top_p": 0.9,
                    "stop": ["\n\n"],  # Stop at double newline
                },
            )

            # Debug: Print raw response
            raw_response = response.get("response", "")
            print(f"    🔍 Raw response: '{raw_response}'")

            if not raw_response or not raw_response.strip():
                print(f"    ⚠️ Empty response received, retrying...")
                if attempt < retry_count - 1:
                    time.sleep(2)
                    continue
                else:
                    description = f"Command: {command}"
            else:
                description = raw_response.strip()
                # Remove any extra newlines and clean up
                description = " ".join(description.split())

                # Remove common prefixes that models add
                prefixes_to_remove = [
                    "Description:",
                    "This command",
                    "The command",
                    "Answer:",
                ]
                for prefix in prefixes_to_remove:
                    if description.startswith(prefix):
                        description = description[len(prefix) :].strip()

            # Print the cleaned explanation
            print(f"    ✓ Explanation: {description}")

            return {
                "source_file": filename,
                "command": command,
                "explanation": description,
            }

        except Exception as e:
            if attempt < retry_count - 1:
                print(
                    f"      ⚠️ Retry {attempt + 1}/{retry_count - 1} due to error: {e}"
                )
                time.sleep(2)
            else:
                print(f"      ❌ Error after {retry_count} attempts: {e}")
                # Fallback: use command itself as description
                return {
                    "source_file": filename,
                    "command": command,
                    "explanation": f"TagTool command: {command}",
                }


def get_command_files(folder_path: str, extensions: List[str] = None) -> List[Path]:
    """
    Get all command files from the specified folder.
    """
    if extensions is None:
        # Default extensions - NOW INCLUDES .cmds!
        extensions = [".cmd", ".cmds", ".txt", ".commands", ".script", ".tagtool"]

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_files = []

    # Search for each extension
    for ext in extensions:
        # Recursive search
        files = list(folder.rglob(f"*{ext}"))
        all_files.extend(files)

    # Remove duplicates and sort
    all_files = sorted(set(all_files))

    if not all_files:
        print(f"Warning: No command files found in {folder_path}")

        # Debug: show what IS in the folder
        print(f"\n📁 Contents of {folder_path}:")
        try:
            for item in folder.iterdir():
                if item.is_file():
                    print(
                        f"  📄 {item.name} (size: {item.stat().st_size} bytes, ext: '{item.suffix}')"
                    )
                elif item.is_dir():
                    print(f"  📁 {item.name}/ (directory)")
        except Exception as e:
            print(f"  Error listing directory: {e}")

        print(f"\n💡 Searched for extensions: {', '.join(extensions)}")

    return all_files


def process_file(file_path: Path, model: str = "gemma3:27b") -> List[Dict]:
    """Process a single file and analyze each command."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {file_path.name}")
    print(f"Full path: {file_path}")
    print(f"{'=' * 60}")

    commands = extract_commands_from_file(file_path)
    print(f"Found {len(commands)} commands in this file")

    if not commands:
        print("⚠️ No commands found in this file")
        return []

    results = []

    for i, command in enumerate(commands, 1):
        result = analyze_single_command(
            command, file_path.name, i, len(commands), model
        )
        results.append(result)

        # Save intermediate results every 10 commands
        if i % 10 == 0:
            intermediate_path = f"intermediate_results_{file_path.stem}.json"
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"    💾 Saved intermediate results to {intermediate_path}")

    print(f"\n✓ Completed {file_path.name}: {len(results)} commands analyzed")
    return results


def save_all_results(results: List[Dict], output_path: str):
    """Save raw results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Raw results saved: {output_path}")


def format_for_instruction_tuning(results: List[Dict], output_path: str):
    """
    Format for instruction-based fine-tuning (Alpaca format)
    Input = explanation, Output = command
    """
    formatted_data = []

    for result in results:
        formatted_data.append(
            {
                "instruction": "Generate the TagTool command for this action:",
                "input": result["explanation"],  # The explanation
                "output": result["command"],  # The actual command
                "metadata": {"source_file": result["source_file"]},
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Instruction format: {output_path}")


def format_for_chat_tuning(results: List[Dict], output_path: str):
    """
    Format for chat-based fine-tuning
    Input = explanation, Output = command
    """
    formatted_data = []

    for result in results:
        formatted_data.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in TagTool commands for Halo modding. Generate the correct TagTool command for the requested action.",
                    },
                    {
                        "role": "user",
                        "content": f"Generate a TagTool command to: {result['explanation']}",
                    },
                    {"role": "assistant", "content": result["command"]},
                ],
                "metadata": {"source_file": result["source_file"]},
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Chat format: {output_path}")


def format_for_jsonl(results: List[Dict], output_path: str):
    """
    Format as JSONL (one JSON per line)
    Input = explanation, Output = command
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            entry = {
                "instruction": "Generate the TagTool command for this action:",
                "input": result["explanation"],
                "output": result["command"],
                "source_file": result["source_file"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  ✓ JSONL format: {output_path}")


def format_for_sharegpt(results: List[Dict], output_path: str):
    """
    Format for ShareGPT/Vicuna style training
    Input = explanation, Output = command
    """
    formatted_data = []

    for result in results:
        formatted_data.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Generate a TagTool command to: {result['explanation']}",
                    },
                    {"from": "gpt", "value": result["command"]},
                ],
                "source": result["source_file"],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ ShareGPT format: {output_path}")


def check_ollama_connection(model_name: str = "gemma3:27b"):
    """Check if Ollama is running and the model is available."""
    try:
        print("Checking Ollama connection...")

        # Try a simple test generation
        try:
            test_response = ollama.generate(
                model=model_name, prompt="Say 'test'", options={"num_predict": 10}
            )
            response_text = test_response.get("response", "").strip()
            print(f"✓ Ollama is responding! Test response: '{response_text}'")

            if not response_text:
                print(
                    "⚠️ Warning: Ollama responded but with empty text. This might cause issues."
                )
                print("   Consider trying a different model.")

            return True
        except Exception as test_error:
            print(f"✗ Connection test failed: {test_error}")

            # Try to list models for additional info
            try:
                models_response = ollama.list()
                if isinstance(models_response, dict):
                    models = models_response.get("models", [])
                else:
                    models = models_response

                print("\nAvailable models:")
                for model in models:
                    if isinstance(model, dict):
                        print(f"  - {model.get('name', model.get('model', 'unknown'))}")
                    else:
                        print(f"  - {model}")
            except:
                pass

            return False

    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure Ollama is running:")
        print("   ollama serve")
        print("2. Pull the model if not already available:")
        print(f"   ollama pull {model_name}")
        print("3. Test Ollama manually:")
        print(f"   ollama run {model_name}")
        return False


def process_folder(
    folder_path: str,
    output_base: str,
    model: str = "gemma3:27b",
    extensions: List[str] = None,
):
    """Process all command files in a folder and generate multiple output formats."""
    print(f"\n{'=' * 60}")
    print(f"Scanning folder: {folder_path}")
    print(f"Absolute path: {os.path.abspath(folder_path)}")
    print(f"{'=' * 60}")

    command_files = get_command_files(folder_path, extensions)

    if not command_files:
        print("\n❌ No command files found!")
        print("\n💡 Try specifying custom extensions:")
        print("   Example: --extensions .txt .script .commands")
        return

    print(f"\nFound {len(command_files)} command file(s):")
    for f in command_files:
        print(f"  📄 {f.name} ({f.stat().st_size} bytes)")

    all_results = []
    start_time = time.time()
    total_commands = 0

    for i, cmd_file in enumerate(command_files, 1):
        print(f"\n{'#' * 60}")
        print(f"FILE {i}/{len(command_files)}")
        print(f"{'#' * 60}")

        file_results = process_file(cmd_file, model)
        all_results.extend(file_results)
        total_commands += len(file_results)

        # Save progress after each file
        progress_file = f"{output_base}_progress.json"
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Progress saved to: {progress_file}")

        elapsed = time.time() - start_time
        avg_time_per_file = elapsed / i
        remaining_files = len(command_files) - i
        remaining_time = avg_time_per_file * remaining_files

        print(f"\n{'=' * 60}")
        print(f"PROGRESS SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Files processed: {i}/{len(command_files)}")
        print(f"  Total commands processed: {total_commands}")
        print(f"  Time elapsed: {elapsed / 60:.1f} minutes")
        if remaining_files > 0:
            print(f"  Estimated time remaining: {remaining_time / 60:.1f} minutes")
        print(f"{'=' * 60}")

    print(f"\n{'#' * 60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'#' * 60}")
    print(f"Total files processed: {len(command_files)}")
    print(f"Total commands analyzed: {len(all_results)}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"{'#' * 60}\n")

    if not all_results:
        print("\n⚠️ No results to save!")
        return

    print("\n{'='*60}")
    print("SAVING OUTPUT FILES")
    print(f"{'=' * 60}\n")

    # Save raw results first
    save_all_results(all_results, f"{output_base}_raw.json")

    # Generate all formats
    format_for_instruction_tuning(all_results, f"{output_base}_instruction.json")
    format_for_chat_tuning(all_results, f"{output_base}_chat.json")
    format_for_jsonl(all_results, f"{output_base}_instruction.jsonl")
    format_for_sharegpt(all_results, f"{output_base}_sharegpt.json")

    # Summary
    summary_path = f"{output_base}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"TagTool Command Analysis Summary\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Total files processed: {len(command_files)}\n")
        f.write(f"Total commands analyzed: {len(all_results)}\n")
        f.write(f"Processing time: {(time.time() - start_time) / 60:.1f} minutes\n")
        f.write(
            f"Average commands per file: {len(all_results) / len(command_files):.1f}\n\n"
        )
        f.write(f"Output formats generated:\n")
        f.write(f"  - {output_base}_raw.json (Raw data)\n")
        f.write(f"  - {output_base}_instruction.json (Alpaca/instruction format)\n")
        f.write(f"  - {output_base}_chat.json (Chat/conversation format)\n")
        f.write(f"  - {output_base}_instruction.jsonl (JSONL format)\n")
        f.write(f"  - {output_base}_sharegpt.json (ShareGPT format)\n\n")
        f.write(f"Commands per file:\n")

        from collections import defaultdict

        by_file = defaultdict(int)
        for result in all_results:
            by_file[result["source_file"]] += 1

        for filename, count in sorted(by_file.items()):
            f.write(f"  {filename}: {count} commands\n")

        # Check for empty explanations
        empty_count = sum(
            1
            for r in all_results
            if not r["explanation"] or r["explanation"].strip() == ""
        )
        if empty_count > 0:
            f.write(f"\n⚠️ Warning: {empty_count} commands have empty explanations\n")

    print(f"\n  ✓ Summary: {summary_path}")

    # Check for issues
    empty_explanations = [
        r for r in all_results if not r["explanation"] or r["explanation"].strip() == ""
    ]
    if empty_explanations:
        print(
            f"\n⚠️ WARNING: {len(empty_explanations)} commands have empty explanations!"
        )
        print(f"   First few examples:")
        for i, result in enumerate(empty_explanations[:5], 1):
            print(f"   {i}. Command: {result['command'][:60]}...")

    print(f"\n{'=' * 60}")
    print(f"FORMAT GUIDE")
    print(f"{'=' * 60}")
    print(f"Use case recommendations:")
    print(f"  • Raw data: _raw.json")
    print(f"  • Llama/Mistral fine-tuning: _instruction.json")
    print(f"  • ChatGPT-style models: _chat.json")
    print(f"  • Vicuna/ShareGPT: _sharegpt.json")
    print(f"  • Streaming/large datasets: _instruction.jsonl")
    print(f"{'=' * 60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze TagTool commands for fine-tuning"
    )
    parser.add_argument(
        "--folder", default="Data", help="Folder containing command files"
    )
    parser.add_argument(
        "--output", default="tagtool_finetuning", help="Output file base name"
    )
    parser.add_argument("--model", default="gemma3:27b", help="Ollama model to use")
    parser.add_argument(
        "--extensions",
        nargs="+",
        help="File extensions to search for (e.g., .cmd .txt)",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    input_folder = args.folder
    output_base = args.output
    model_name = args.model
    extensions = args.extensions

    print(f"\n{'=' * 60}")
    print(f"TagTool Command Analyzer for Fine-Tuning")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Input folder: {input_folder}")
    print(f"Output base: {output_base}")
    if extensions:
        print(f"Extensions: {', '.join(extensions)}")
    print(f"Mode: INDIVIDUAL COMMANDS (line-by-line)")
    print(f"{'=' * 60}\n")

    # Check Ollama connection
    if not check_ollama_connection(model_name):
        print("\n❌ Cannot proceed without Ollama connection.")
        print("\nQuick start:")
        print("1. Start Ollama: ollama serve")
        print(f"2. Pull model: ollama pull {model_name}")
        print("3. Run this script again")
        return

    if not os.path.exists(input_folder):
        print(f"\n❌ Error: Folder '{input_folder}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Please create the folder and add your command files.")
        return

    # Count total commands for estimate
    print("\nCounting commands in all files...")
    command_files = get_command_files(input_folder, extensions)

    if not command_files:
        print("\n❌ No command files found to process!")
        return

    total_commands = 0
    for cmd_file in command_files:
        commands = extract_commands_from_file(cmd_file)
        total_commands += len(commands)
        print(f"  {cmd_file.name}: {len(commands)} commands")

    print(f"\n{'=' * 60}")
    print(f"PROCESSING ESTIMATE")
    print(f"{'=' * 60}")
    print(f"Total files: {len(command_files)}")
    print(f"Total commands to analyze: {total_commands}")
    print(
        f"Estimated time (at ~2 sec/command): {(total_commands * 2) / 60:.1f} minutes"
    )
    print(f"{'=' * 60}\n")

    if not args.yes:
        response = input("Continue? (y/n): ")
        if response.lower() != "y":
            print("Cancelled.")
            return

    print("\n✓ Starting processing...\n")

    process_folder(input_folder, output_base, model_name, extensions)


if __name__ == "__main__":
    main()
