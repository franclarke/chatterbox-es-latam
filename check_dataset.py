#!/usr/bin/env python3
"""
Utility to print dataset statistics: number of examples, average duration,
distribution by nationality, percentage of AR (Argentina), etc.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def load_dataset(dataset_path: str) -> list:
    """Load dataset from JSON or JSONL file."""
    path = Path(dataset_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both list and dict with 'data' key
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            else:
                raise ValueError("JSON file must contain a list or a dict with 'data' key")
    
    elif path.suffix == ".jsonl":
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .jsonl")


def calculate_statistics(examples: list) -> dict:
    """Calculate statistics from the dataset examples."""
    stats = {
        "total_examples": len(examples),
        "duration_stats": None,
        "nationality_distribution": None,
        "ar_percentage": None,
    }
    
    if not examples:
        return stats
    
    # Calculate duration statistics
    durations = []
    for ex in examples:
        duration = ex.get("duration") or ex.get("duracion") or ex.get("duration_seconds")
        if duration is not None:
            try:
                durations.append(float(duration))
            except (TypeError, ValueError):
                pass
    
    if durations:
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        stats["duration_stats"] = {
            "count": len(durations),
            "total_seconds": total_duration,
            "average_seconds": avg_duration,
            "min_seconds": min_duration,
            "max_seconds": max_duration,
        }
    
    # Calculate nationality distribution
    nationalities = []
    for ex in examples:
        nationality = (
            ex.get("nationality")
            or ex.get("nacionalidad")
            or ex.get("country")
            or ex.get("pais")
        )
        if nationality:
            nationalities.append(nationality.upper())
    
    if nationalities:
        nationality_counts = Counter(nationalities)
        total_with_nationality = len(nationalities)
        stats["nationality_distribution"] = {
            nat: {
                "count": count,
                "percentage": (count / total_with_nationality) * 100,
            }
            for nat, count in sorted(nationality_counts.items(), key=lambda x: -x[1])
        }
        
        # Calculate AR percentage specifically
        ar_count = nationality_counts.get("AR", 0)
        stats["ar_percentage"] = (ar_count / total_with_nationality) * 100 if total_with_nationality > 0 else 0.0
    
    return stats


def print_statistics(stats: dict) -> None:
    """Print statistics in a readable format."""
    print("=" * 60)
    print("DATASET STATISTICS / ESTADÍSTICAS DEL DATASET")
    print("=" * 60)
    
    # Total examples
    print(f"\n📊 Total examples / Cantidad de ejemplos: {stats['total_examples']}")
    
    # Duration stats
    if stats["duration_stats"]:
        ds = stats["duration_stats"]
        print(f"\n⏱️  Duration statistics / Estadísticas de duración:")
        print(f"   Examples with duration: {ds['count']}")
        print(f"   Total duration: {ds['total_seconds']:.2f} seconds ({ds['total_seconds']/3600:.2f} hours)")
        print(f"   Average duration / Duración media: {ds['average_seconds']:.2f} seconds")
        print(f"   Min duration: {ds['min_seconds']:.2f} seconds")
        print(f"   Max duration: {ds['max_seconds']:.2f} seconds")
    else:
        print("\n⏱️  Duration statistics: No duration data available")
    
    # Nationality distribution
    if stats["nationality_distribution"]:
        print(f"\n🌎 Nationality distribution / Distribución por nacionalidad:")
        for nat, data in stats["nationality_distribution"].items():
            print(f"   {nat}: {data['count']} ({data['percentage']:.1f}%)")
        
        # AR percentage highlighted
        if stats["ar_percentage"] is not None:
            print(f"\n🇦🇷 AR percentage / Porcentaje de AR: {stats['ar_percentage']:.1f}%")
    else:
        print("\n🌎 Nationality distribution: No nationality data available")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Print dataset statistics: number of examples, average duration, "
                    "distribution by nationality, percentage of AR, etc."
    )
    parser.add_argument(
        "dataset",
        help="Path to the dataset file (.json or .jsonl)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output statistics as JSON instead of human-readable format"
    )
    
    args = parser.parse_args()
    
    try:
        examples = load_dataset(args.dataset)
        stats = calculate_statistics(examples)
        
        if args.json:
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        else:
            print_statistics(stats)
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in dataset file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
