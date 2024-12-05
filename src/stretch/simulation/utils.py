import pkg_resources

models_path = pkg_resources.resource_filename("stretch", "simulation/models")
default_scene_xml_path = models_path + "/scene.xml"

def get_default_scene_path() -> str:
    """Return the default scene.xml path."""
    return default_scene_xml_path
