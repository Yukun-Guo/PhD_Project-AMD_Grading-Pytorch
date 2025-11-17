#!/usr/bin/env python3
"""
Quick verification script to check if the CAMVis summary visualizations
have been properly fixed and contain data in both panels.
"""

import json
from pathlib import Path


def verify_fix(session_dir: str = "CAMVis/session_20251114_172306"):
    """Verify that the CAMVis summary fix was successful."""
    session_path = Path(session_dir)
    
    print(f"🔍 Verifying CAMVis fix for session: {session_path}")
    
    # Check if summary images exist
    summaries_dir = session_path / "summaries"
    analysis_summary = summaries_dir / "analysis_summary.png"
    detailed_analysis = summaries_dir / "heatmap_detailed_analysis.png"
    
    print(f"📊 Analysis summary exists: {analysis_summary.exists()}")
    print(f"📈 Detailed analysis exists: {detailed_analysis.exists()}")
    
    # Check the statistics file
    stats_file = session_path / "reports" / "overall_statistics.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"📋 Statistics loaded successfully")
        print(f"📊 Total samples: {stats['overall']['total_samples_processed']}")
        print(f"🎯 Overall accuracy: {stats['overall']['overall_accuracy']:.1%}")
        
        # Check heatmap summaries
        heatmap_data_found = False
        for split_name, split_data in stats['splits'].items():
            heatmap_summary = split_data.get('heatmap_summary', {})
            if heatmap_summary:
                heatmap_data_found = True
                print(f"✅ {split_name} split has heatmap data for {len(heatmap_summary)} input types")
                
                for inp_name, inp_stats in heatmap_summary.items():
                    sample_count = inp_stats.get('sample_count', 0)
                    avg_max = inp_stats.get('avg_max_activation', 0)
                    print(f"   - {inp_name.upper()}: {sample_count} samples, avg_max={avg_max:.3f}")
            else:
                print(f"❌ {split_name} split has empty heatmap data")
        
        if heatmap_data_found:
            print("✅ SUCCESS: Heatmap statistics are properly populated!")
            print("✅ Both panels in the visualizations should now contain data")
        else:
            print("❌ FAILURE: Heatmap statistics are still empty")
    else:
        print("❌ Statistics file not found")
    
    print(f"\n📁 Check the visualizations at:")
    print(f"   - {analysis_summary}")
    print(f"   - {detailed_analysis}")


if __name__ == '__main__':
    verify_fix()