#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_obj_loader.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
	int loadObject(string filename);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;

	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> tinyMaterials;
	tinyobj::attrib_t attrib;
	std::string err;
    RenderState state;
};
