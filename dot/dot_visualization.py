from graphviz import Source
import os
import re


dot_path = "./runs/prototree_face/run_54/pruned_and_projected/treevis.dot"

# dot_path = path + "/treevis.dot"


def clean_dot_content(dot_content):
    # Normalize all backslashes to forward slashes first
    dot_content = dot_content.replace("\\", "/")
    print(dot_content)

    # Remove the long hyperparameter path segment
    
    # to_remove = r'/runs/prototree_face/run_10'
    
    # dot_content = re.sub(to_remove, 'best_valid_model', dot_content)

    # Remove the pruning path segment
    # to_remove = r'/first_prune_siblings/first_pruned_and_projected/second_prune_siblings'
    # dot_content = re.sub(to_remove, '', dot_content)

    # Fix local_viz_test -> local_viz
    # dot_content = dot_content.replace("local_viz_test", "local_viz")

    return dot_content

with open(dot_path, 'r', encoding='utf-8') as f:
            dot_content = f.read()

dot_content = clean_dot_content(dot_content)
src = Source(dot_content)
src.render("treevis", format="pdf")



# # --- Render local visualizations ---
# local_viz_path = os.path.join(path, "local_viz_test")
# i = 0

# for root, dirs, files in os.walk(local_viz_path):
#     for folder_name in sorted(dirs):  # sorted for deterministic ordering
#         print(folder_name)
#         dot_file = os.path.join(local_viz_path, folder_name, "predvis_0.dot")

#         if not os.path.exists(dot_file):
#             print(f"  Skipping {folder_name}: predvis_0.dot not found")
#             continue

#         with open(dot_file, 'r', encoding='utf-8') as f:
#             dot_content = f.read()

#         dot_content = clean_dot_content(dot_content)
#         src = Source(dot_content)
#         src.render(f"local_{i}", format="pdf")
#         i += 1
