#include "triangle.h"

#include "CGL/CGL.h"
#include "GL/glew.h"

namespace CGL { namespace StaticScene {

Triangle::Triangle(const Mesh* mesh, size_t v1, size_t v2, size_t v3) :
    mesh(mesh), v1(v1), v2(v2), v3(v3) { }

BBox Triangle::get_bbox() const {

  Vector3D p1(mesh->positions[v1]), p2(mesh->positions[v2]), p3(mesh->positions[v3]);
  BBox bb(p1);
  bb.expand(p2);
  bb.expand(p3);
  return bb;

}

bool Triangle::intersect(const Ray& r) const {
  return intersect(r, nullptr);
}

bool Triangle::intersect(const Ray& r, Intersection *isect) const {
  Vector3D p0 = mesh->positions[v1],
           p1 = mesh->positions[v2],
           p2 = mesh->positions[v3],
           n0(mesh->normals[v1]),
           n1(mesh->normals[v2]),
           n2(mesh->normals[v3]),
           e1 = p1 - p0,
           e2 = p2 - p0,
           s = r.o - p0,
           s1 = cross(r.d, e2),
           s2 = cross(s, e1),
           res(dot(s2, e2), dot(s1, s), dot(s2, r.d));
  res /= dot(s1, e1);
  const double &t = res.x,
               &b1 = res.y,
               &b2 = res.z,
               b0 = 1 - b1 - b2;
  bool ret = r.min_t <= t && t <= r.max_t && b0 >= 0 && b1 >= 0 && b2 >= 0;
  if (ret) {
    r.max_t = t;
    if (isect) {
      isect->t = t;
      isect->n = b0 * n0 + b1 * n1 + b2 * n2;
      isect->primitive = this;
      isect->bsdf = get_bsdf();
    }
  }
  return ret;
}

void Triangle::draw(const Color& c, float alpha) const {
  glColor4f(c.r, c.g, c.b, alpha);
  glBegin(GL_TRIANGLES);
  glVertex3d(mesh->positions[v1].x,
             mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x,
             mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x,
             mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}

void Triangle::drawOutline(const Color& c, float alpha) const {
  glColor4f(c.r, c.g, c.b, alpha);
  glBegin(GL_LINE_LOOP);
  glVertex3d(mesh->positions[v1].x,
             mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x,
             mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x,
             mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}



} // namespace StaticScene
} // namespace CGL
