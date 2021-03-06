#include "bvh.h"

#include "CGL/CGL.h"
#include "static_scene/triangle.h"
#include "static_scene/blackhole.h"

#include <iostream>
#include <stack>

using namespace std;

namespace CGL { namespace StaticScene {

BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {

  root = construct_bvh(_primitives, max_leaf_size);

}

BVHAccel::~BVHAccel() {
  if (root) delete root;
}

BBox BVHAccel::get_bbox() const {
  return root->bb;
}

void BVHAccel::draw(BVHNode *node, const Color& c, float alpha) const {
  if (node->isLeaf()) {
    for (Primitive *p : *(node->prims))
      p->draw(c, alpha);
  } else {
    draw(node->l, c, alpha);
    draw(node->r, c, alpha);
  }
}

void BVHAccel::drawOutline(BVHNode *node, const Color& c, float alpha) const {
  if (node->isLeaf()) {
    for (Primitive *p : *(node->prims))
      p->drawOutline(c, alpha);
  } else {
    drawOutline(node->l, c, alpha);
    drawOutline(node->r, c, alpha);
  }
}

BVHNode *BVHAccel::construct_bvh(const std::vector<Primitive*>& prims, size_t max_leaf_size) {
  BBox bbox;
  for (Primitive *p : prims)
    bbox.expand(p->get_bbox());
  BVHNode *node = new BVHNode(bbox);
  if (prims.size() <= max_leaf_size
#if ACCEL == 0
      || true
#endif // ACCEL
     ) {
    node->prims = new vector<Primitive *>(prims);
  } else {
    vector<Primitive *> prims_l, prims_r;
    if (bbox.extent.x > bbox.extent.y && bbox.extent.x > bbox.extent.z) {
      double c = bbox.centroid().x;
      for (Primitive *p : prims)
        if (p->get_bbox().centroid().x < c)
          prims_l.push_back(p);
        else
          prims_r.push_back(p);
    } else if (bbox.extent.y > bbox.extent.x && bbox.extent.y > bbox.extent.z) {
      double c = bbox.centroid().y;
      for (Primitive *p : prims)
        if (p->get_bbox().centroid().y < c)
          prims_l.push_back(p);
        else
          prims_r.push_back(p);
    } else {
      double c = bbox.centroid().z;
      for (Primitive *p : prims)
        if (p->get_bbox().centroid().z < c)
          prims_l.push_back(p);
        else
          prims_r.push_back(p);
    }
    if (prims_l.empty() || prims_r.empty()) {
      prims_l.clear();
      prims_r.clear();
      for (size_t i = 0; i < prims.size() / 2; ++i)
        prims_l.push_back(prims[i]);
      for (size_t i = prims.size() / 2; i < prims.size(); ++i)
        prims_r.push_back(prims[i]);
    }
    node->l = construct_bvh(prims_l, max_leaf_size);
    node->r = construct_bvh(prims_r, max_leaf_size);
  }
  return node;
}


bool BVHAccel::intersect(const Ray& ray, BVHNode *node) const {
  return intersect(ray, nullptr, node);
}

bool BVHAccel::intersect(const Ray& ray, Intersection* i, BVHNode *node) const {
  Ray micro_ray(ray.o, ray.d, 0.0);
  for (int j = 0; j * global_black_hole.delta_theta < 2 * M_PI; ++j) {
    micro_ray = global_black_hole.next_micro_ray(micro_ray);
    if (global_black_hole.intersect(micro_ray))
      return false;
    if (intersect_micro(micro_ray, i, node))
      return true;
  }
  return false;
}

bool BVHAccel::intersect_micro(const Ray& ray, Intersection* i, BVHNode *node) const {
  double _;
  if (!node->bb.intersect(ray, _, _))
    return false;
  bool hit = false;
  if (node->prims) {
    for (Primitive *p : *(node->prims)) {
      ++total_isects;
      if (i) {
        if (p->intersect(ray, i))
          hit = true;
      } else {
        if (p->intersect(ray))
          hit = true;
      }
    }
  } else {
    if (intersect_micro(ray, i, node->l))
      hit = true;
    if (intersect_micro(ray, i, node->r))
      hit = true;
  }
  return hit;
}

}  // namespace StaticScene
}  // namespace CGL
