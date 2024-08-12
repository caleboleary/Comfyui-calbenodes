import os
import importlib.util
import inspect

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def import_nodes_from_dir(dir_path):
    for filename in os.listdir(dir_path):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove .py extension
            file_path = os.path.join(dir_path, filename)
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and hasattr(obj, 'CATEGORY') and obj.CATEGORY == "calbenodes":
                        NODE_CLASS_MAPPINGS[name] = obj
                        # You can customize this to set a different display name if needed
                        NODE_DISPLAY_NAME_MAPPINGS[name] = name.replace('Node', '').replace('_', ' ').title()
            except Exception as e:
                print(f"Error importing {filename}: {e}")

# Assuming this __init__.py is in the 'ComfyUI-calbenodes' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
nodes_dir = os.path.join(current_dir, 'nodes')
import_nodes_from_dir(nodes_dir)

# Print discovered nodes for debugging
print("Discovered nodes:")
for name, cls in NODE_CLASS_MAPPINGS.items():
    print(f"  {name}: {cls}")