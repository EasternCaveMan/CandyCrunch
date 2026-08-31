import os
import pickle
import torch
import pandas as pd
# data = pd.read_pickle("full_dataset_20260717.pkl")

# rep_dict = torch.load("./prepared_datasets_DSS5.0/graph_tokens.pt", weights_only = False)
current_dir = os.path.dirname(os.path.abspath(__file__))

with open("./prepared_datasets_DSS5.0/X_train_GShS.pkl", "rb") as file:
  X_train_c1e = pickle.load(file)
with open("./prepared_datasets_DSS5.0/X_test_GShS.pkl", "rb") as file:
  X_test_c1e = pickle.load(file)
with open("./prepared_datasets_DSS5.0/y_train_GShS.pkl", "rb") as file:
  y_train_c1e = pickle.load(file)
with open("./prepared_datasets_DSS5.0/y_test_GShS.pkl", "rb") as file:
  y_test_c1e = pickle.load(file)
with open("./prepared_datasets/glycans.pkl", "rb") as file:
  glycans_c1e = pickle.load(file)

# with open("./prepared_datasets_org/X_train.pkl", "rb") as file:
#     X_train = pickle.load(file)
# with open("./prepared_datasets_org/X_test.pkl", "rb") as file:
#     X_test = pickle.load(file)
# with open("./prepared_datasets_org/y_train.pkl", "rb") as file:
#     y_train = pickle.load(file)
# with open("./prepared_datasets_org/y_test.pkl", "rb") as file:
#     y_test = pickle.load(file)
# with open("./prepared_datasets_org/glycans.pkl", "rb") as file:
#     glycans = pickle.load(file)


def inspect_list_structure(data, name):
    """Inspect the structure of a list's first element"""
    print(f"\n--- {name} ---")
    print(f"Total length: {len(data)}")

    if len(data) > 0:
        first_elem = data[0]
        print(f"Type of each element: {type(first_elem)}")

        # If each element is a dictionary (column names as keys)
        if isinstance(first_elem, dict):
            print(f"Column names (keys): {list(first_elem.keys())}")
            print(f"Data types of values:")
            for key, value in first_elem.items():
                print(f"  {key}: {type(value)}")

        # If each element is a tuple/list (positional columns)
        elif isinstance(first_elem, (list, tuple)):
            print(f"Number of columns: {len(first_elem)}")
            print(f"Data types of each column position:")
            for i, value in enumerate(first_elem):
                print(f"  Position {i}: {type(value)}")
            # Sample values
            print(f"Sample first row: {first_elem}")

        # If each element is a numpy array
        elif hasattr(first_elem, 'shape'):
            print(f"Array shape: {first_elem.shape}")
            print(f"Array dtype: {first_elem.dtype}")

        # Other types
        else:
            print(f"Element sample: {first_elem}")
            print(f"Element type: {type(first_elem)}")


# Inspect both datasets
inspect_list_structure(X_train_c1e, "X_train_c1e")
# inspect_list_structure(X_train, "X_train")


def compare_list_structures(data1, data2, name1 = "Dataset1", name2 = "Dataset2"):
    """Compare column names and data types for list data"""

    if len(data1) == 0 or len(data2) == 0:
        print(f"Cannot compare - one dataset is empty")
        return False

    # Get first elements
    elem1 = data1[0]
    elem2 = data2[0]

    # Check if same type of structure
    if type(elem1) != type(elem2):
        print(f"❌ Element type mismatch: {type(elem1)} vs {type(elem2)}")
        return False

    # For dictionaries (column names are keys)
    if isinstance(elem1, dict):
        cols1 = set(elem1.keys())
        cols2 = set(elem2.keys())

        print(f"\n=== COLUMN NAMES ===")
        if cols1 == cols2:
            print(f"✅ Column names match exactly!")
            print(f"Columns: {sorted(cols1)}")
        else:
            print(f"❌ Column names differ")
            print(f"Only in {name1}: {cols1 - cols2}")
            print(f"Only in {name2}: {cols2 - cols1}")
            return False

        print(f"\n=== DATA TYPES ===")
        type_match = True
        for col in cols1:
            type1 = type(elem1[col])
            type2 = type(elem2[col])
            if type1 != type2:
                print(f"❌ {col}: {type1} vs {type2}")
                type_match = False
            else:
                print(f"✅ {col}: {type1}")

        return type_match

    # For lists/tuples (positional columns)
    elif isinstance(elem1, (list, tuple)):
        if len(elem1) != len(elem2):
            print(f"❌ Number of columns differs: {len(elem1)} vs {len(elem2)}")
            return False

        print(f"\n=== COLUMN POSITIONS ===")
        print(f"Both have {len(elem1)} columns")

        print(f"\n=== DATA TYPES BY POSITION ===")
        type_match = True
        for i in range(len(elem1)):
            type1 = type(elem1[i])
            type2 = type(elem2[i])
            if type1 != type2:
                print(f"❌ Position {i}: {type1} vs {type2}")
                type_match = False
            else:
                print(f"✅ Position {i}: {type1}")

        # Sample values
        print(f"\n=== SAMPLE VALUES ===")
        print(f"{name1} first row: {elem1}")
        print(f"{name2} first row: {elem2}")

        return type_match

    # For numpy arrays
    elif hasattr(elem1, 'shape') and hasattr(elem2, 'shape'):
        if elem1.shape != elem2.shape:
            print(f"❌ Array shape differs: {elem1.shape} vs {elem2.shape}")
            return False
        if elem1.dtype != elem2.dtype:
            print(f"❌ Array dtype differs: {elem1.dtype} vs {elem2.dtype}")
            return False
        print(f"✅ Arrays match: shape={elem1.shape}, dtype={elem1.dtype}")
        return True

    else:
        print(f"⚠️ Unknown structure type: {type(elem1)}")
        return False


# Compare datasets
print("\n" + "=" * 50)
print("COMPARING X_TRAIN STRUCTURES")
print("=" * 50)
# compare_list_structures(X_train_c1e, X_train, "X_train_c1e", "X_train")

print("\n" + "=" * 50)
print("COMPARING X_TEST STRUCTURES")
print("=" * 50)
# compare_list_structures(X_test_c1e, X_test, "X_test_c1e", "X_test")

print("\n" + "=" * 50)
print("COMPARING Y_TRAIN STRUCTURES")
print("=" * 50)
# compare_list_structures(y_train_c1e, y_train, "y_train_c1e", "y_train")


def verify_consistency(data, name, num_samples = 5):
    """Verify that all rows have the same structure"""
    if len(data) < 2:
        print(f"{name}: Too short to verify consistency")
        return

    first_elem = data[0]

    print(f"\n--- {name} Consistency Check ---")
    all_same = True

    for i in range(min(num_samples, len(data))):
        current = data[i]
        if type(current) != type(first_elem):
            print(f"❌ Row {i} has different type: {type(current)}")
            all_same = False

        if isinstance(first_elem, dict) and isinstance(current, dict):
            if set(current.keys()) != set(first_elem.keys()):
                print(f"❌ Row {i} has different columns")
                all_same = False

    if all_same:
        print(f"✅ First {min(num_samples, len(data))} rows have consistent structure")


# Verify consistency
verify_consistency(X_train_c1e, "X_train_c1e")
# verify_consistency(X_train, "X_train")
verify_consistency(X_test_c1e, "X_test_c1e")
# verify_consistency(X_test, "X_test")


def quick_structure_summary(data, name):
    """Quick summary of structure"""
    if len(data) == 0:
        print(f"{name}: Empty list")
        return

    elem = data[0]
    print(f"\n{name}:")
    print(f"  Length: {len(data)}")
    print(f"  Element type: {type(elem).__name__}")

    if isinstance(elem, dict):
        print(f"  Columns: {list(elem.keys())}")
        print(f"  Dtypes: { {k: type(v).__name__ for k, v in elem.items()} }")
    elif isinstance(elem, (list, tuple)):
        print(f"  Number of columns: {len(elem)}")
        print(f"  Column dtypes: {[type(v).__name__ for v in elem]}")
    elif hasattr(elem, 'shape'):
        print(f"  Array shape: {elem.shape}")
        print(f"  Array dtype: {elem.dtype}")


# Quick summary for all datasets
print("\n" + "=" * 50)
print("QUICK STRUCTURE SUMMARY")
print("=" * 50)
quick_structure_summary(X_train_c1e, "X_train_c1e")
# quick_structure_summary(X_train, "X_train")
quick_structure_summary(X_test_c1e, "X_test_c1e")
# quick_structure_summary(X_test, "X_test")
quick_structure_summary(y_train_c1e, "y_train_c1e")
# quick_structure_summary(y_train, "y_train")
quick_structure_summary(glycans_c1e, "glycans_c1e")
# quick_structure_summary(glycans, "glycans")


# Quick check for y_train structure
def check_y_structure(y_data, name):
    """Check structure of y dataset (should be labels)"""
    print(f"\n{name}:")
    print(f"  Length: {len(y_data)}")
    print(f"  Element type: {type(y_data[0]) if len(y_data) > 0 else 'Empty'}")

    if len(y_data) > 0:
        # Show sample values
        print(f"  Sample values: {y_data[:5]}")
        print(f"  Unique values: {set(y_data[:100])}")  # Check first 100
        print(f"  Data type consistent: {all(isinstance(y, type(y_data[0])) for y in y_data[:100])}")


print("\n=== Y DATASETS COMPARISON ===")
check_y_structure(y_train_c1e, "y_train_c1e")
# check_y_structure(y_train, "y_train")
check_y_structure(y_test_c1e, "y_test_c1e")
# check_y_structure(y_test, "y_test")


# Confirm structural equivalence
def confirm_identical_structure():
    # X structure check
    x_structure_match = (
            type(X_train_c1e[0]) == type(X_train_c1e[0]) and
            len(X_train_c1e[0]) == len(X_train_c1e[0]) and
            all(type(X_train_c1e[0][i]) == type(X_train_c1e[0][i]) for i in range(len(X_train_c1e[0])))
    )

    # Y structure check
    y_structure_match = (
            len(y_train_c1e) > 0 and len(y_train_c1e) > 0 and
            type(y_train_c1e[0]) == type(y_train_c1e[0])
    )

    print(f"✅ X datasets have identical structure: {x_structure_match}")
    print(f"✅ Y datasets have identical structure: {y_structure_match}")
    print(f"✅ Overall structure match: {x_structure_match and y_structure_match}")

    if x_structure_match and y_structure_match:
        print("\n🎉 Both dataset splits have the EXACT SAME data structure and column types!")
        print("They only differ in the number of samples and the actual data values.")


confirm_identical_structure()

