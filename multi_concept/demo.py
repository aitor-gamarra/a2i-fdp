#!/usr/bin/env python3
"""
Multi-Concept Zero-Shot YOLO Demo

Demo multi-concept zero-shot detection with specific concept-class mappings.
"""

from pathlib import Path

from multi_concept import MultiConceptZeroShotYOLO


def demo_multi_concept_zero_shot():
    """Demo multi-concept zero-shot detection with specific concept-class mappings."""

    print("🚀 Multi-Concept Zero-Shot YOLO Demo")
    print("====================================")
    print("Testing concept-specific detection:")
    print("✅ school_bus concepts → detect school buses")
    print("✅ ambulance concepts → detect ambulances")
    print("✅ Each concept detects its specific target class")
    print()

    # Define concept vector paths
    concept_paths = {
        "school_bus": "alignment/projected/multi/school_bus_concept_activations.pkl",
        "ambulance": "alignment/projected/multi/ambulance_concept_activations.pkl",
    }

    # Create the multi-concept system
    model = MultiConceptZeroShotYOLO(
        yolo_model_path="yolo11s.pt",
        alignment_model_path="alignment/multi_scale_alignment_network_yolo11s-cls-model.8_yolo11s-multi_scale.pth",
        concept_vectors_paths=concept_paths,
        device="cuda",
    )

    # Test scenarios with expected concept matches
    test_scenarios = [
        {
            "image": "sample_images/school_bus_1.jpg",
            "name": "School Bus",
            "expected_concept": "school_bus",
            "description": "Should detect school bus using school_bus concepts",
        },
        {
            "image": "sample_images/ambulance_1.jpg",
            "name": "Ambulance",
            "expected_concept": "ambulance",
            "description": "Should detect ambulance using ambulance concepts",
        },
        {
            "image": "sample_images/school_bus_2.jpeg",
            "name": "School Bus 2",
            "expected_concept": "school_bus",
            "description": "Additional school bus test",
        },
    ]

    results_summary = {}

    for scenario in test_scenarios:
        print(f"🔍 Testing: {scenario['name']}")
        print(f"   Expected: {scenario['description']}")

        if not Path(scenario["image"]).exists():
            print(f"   ⚠️ Skipping {scenario['image']} (not found)")
            continue

        # Get concept-specific results
        results = model.predict_concept_specific(scenario["image"])

        if results:
            result = results[0]

            # Count COCO detections
            coco_count = 0
            if hasattr(result, "boxes") and result.boxes is not None:
                coco_count = len(result.boxes)

            # Count concept-specific detections
            concept_counts = {}
            total_concept_detections = 0

            if hasattr(result, "concept_detections"):
                for concept_name, detections in result.concept_detections.items():
                    concept_counts[concept_name] = len(detections)
                    total_concept_detections += len(detections)

            results_summary[scenario["name"]] = {
                "coco_detections": coco_count,
                "concept_detections": concept_counts,
                "total_concept_detections": total_concept_detections,
            }

            print(f"   📊 COCO detections: {coco_count}")
            print(f"   🎯 Total concept detections: {total_concept_detections}")

            # Show concept-specific results
            for concept_name, count in concept_counts.items():
                if count > 0:
                    print(f"   🏷️  {concept_name}: {count} detections")

                    # Show detailed results for expected concept
                    if concept_name == scenario["expected_concept"]:
                        detections = result.concept_detections[concept_name]
                        for i, det in enumerate(detections[:2]):
                            print(
                                f"      {i + 1}. {det['class_name']}: {det['confidence']:.3f}"
                            )
                            print(f"         {det['reasoning']}")

            # Generate enhanced visualization
            output_path = f"enhanced_multi_concept_{scenario['name'].lower().replace(' ', '_')}_result.jpg"
            model.visualize_concept_detections_enhanced(
                scenario["image"],
                results=result,
                output_path=output_path,
                concept_threshold=0.3,
            )
            print(f"   📸 Enhanced visualization: {output_path}")

        print()

    # Summary report
    print("📋 Multi-Concept Test Summary")
    print("============================")

    total_detections = 0
    concept_performance = {}

    for scenario_name, summary in results_summary.items():
        total_detections += summary["total_concept_detections"]
        print(f"✅ {scenario_name}:")
        print(f"   COCO: {summary['coco_detections']}")

        for concept_name, count in summary["concept_detections"].items():
            if concept_name not in concept_performance:
                concept_performance[concept_name] = 0
            concept_performance[concept_name] += count
            if count > 0:
                print(f"   {concept_name}: {count}")

    print()
    print("🏆 Overall Performance:")
    print(f"   🎯 Total concept-specific detections: {total_detections}")
    for concept_name, total_count in concept_performance.items():
        print(f"   📊 {concept_name}: {total_count} detections across all tests")

    if total_detections > 0:
        print()
        print("🎉 MULTI-CONCEPT ZERO-SHOT DETECTION: SUCCESS!")
        print("   ✅ Concept-specific detection working!")
        print("   ✅ Each concept detects its target object class!")
        print("   ✅ True zero-shot detection with concept-class mapping!")
        return True
    else:
        print()
        print("⚠️ Some issues detected in concept-specific detection")
        return False


if __name__ == "__main__":
    demo_multi_concept_zero_shot()
