#pragma warning(disable:4996)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <Python.h>

#define dllExport extern "C" __declspec(dllexport)

#ifndef M_PI
#define M_PI 3.141592653589793238
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef size_t sz;

#define max_val(x, y) ( (x) > (y) ? (x) : (y) )
#define min_val(x, y) ( (x) < (y) ? (x) : (y) )

#define arr_len(arr) ( sizeof(arr) / sizeof*(arr) )

static float
clamp(float t, float min, float max)
{
	return max_val(min, min_val(max, t));
}

static float
clamp01(float t)
{
	return clamp(t, 0.0f, 1.0f);
}

static float
lerp(float x, float y, float t)
{
	return (1.0f - t) * x + t * y;
}

union Vec3f {
	struct {
		float x, y, z;
	};

	struct {
		float r, g, b;
	};

	float el[3];
};

static Vec3f
vec3f(float x, float y, float z)
{
	Vec3f v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}

static Vec3f o3f = vec3f(0.0f, 0.0f, 0.0f);
static Vec3f z3f = vec3f(0.0f, 0.0f, 1.0f);


static Vec3f
operator + (Vec3f a, Vec3f b)
{
	return vec3f(a.x + b.x, a.y + b.y, a.z + b.z);
}

static Vec3f
operator - (Vec3f a, Vec3f b)
{
	return vec3f(a.x - b.x, a.y - b.y, a.z - b.z);
}

static Vec3f
operator - (Vec3f v)
{
	return vec3f(-v.x, -v.y, -v.z);
}

static Vec3f &
operator += (Vec3f &a, Vec3f b)
{
	a = a + b;
	return a;
}

static Vec3f &
operator -= (Vec3f &a, Vec3f b)
{
	a = a - b;
	return a;
}

static Vec3f
operator * (float f, Vec3f v)
{
	return vec3f(f * v.x, f * v.y, f * v.z);
}

static Vec3f
operator * (Vec3f v, float f)
{
	return f * v;
}

static Vec3f &
operator *= (Vec3f &a, float f)
{
	a = a * f;
	return a;
}

static Vec3f
operator / (Vec3f v, float f)
{
	return (1.0f / f) * v;
}

static float
dot(Vec3f a, Vec3f b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3f
cross(Vec3f a, Vec3f b)
{
	return vec3f(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x);
}

static float
squared_norm(Vec3f v)
{
	return dot(v, v);
}

static float
norm(Vec3f v)
{
	return sqrtf(squared_norm(v));
}

static Vec3f
normalized(Vec3f v)
{
	return v / norm(v);
}

static Vec3f
lerp(Vec3f x, Vec3f y, float t)
{
	return vec3f(lerp(x.x, y.x, t), lerp(x.y, y.y, t), lerp(x.z, y.z, t));
}

// RGBA 8 bits per channel.
struct Image {
	i32 width;
	i32 height;

	u8* data;
};

struct Camera {
	Vec3f origin;

	// Unit vectors.
	Vec3f forward;
	Vec3f up;
	Vec3f right;

	float tan_half_field_of_view;
};

struct Material {
	Vec3f color;
};

struct Plane {
	Vec3f origin;
	// Unit vector.
	Vec3f normal;

	i32 material_index;
};

struct Sphere {
	Vec3f center;
	float radius;

	i32 material_index;
};

struct PointLight {
	Vec3f position;
	float intensity;
};

struct Scene {
	Camera camera;

	Vec3f background_color;

	i32 num_planes;
	Plane *planes;

	i32 num_spheres;
	Sphere *spheres;

	i32 num_materials;
	Material *materials;

	i32 num_point_lights;
	PointLight *point_lights;
};

struct Ray {
	Vec3f origin;
	// Unit vector.
	Vec3f direction;
};

static void
init_image(Image *image, i32 width, i32 height)
{
	image->width = width;
	image->height = height;

	image->data = (u8 *)malloc(width * height * 4);
}

static void
terminate_image(Image *image)
{
	free(image->data);
}

static u8*
get_image_pixel(Image *image, i32 x, i32 y)
{
	return image->data + 4 * (y * image->width + x);
}

static void
init_camera(Camera *camera, float field_of_view, Vec3f origin, Vec3f look_at)
{
	camera->tan_half_field_of_view = tanf(0.5f * field_of_view);
	camera->origin = origin;

	camera->forward = normalized(look_at - origin);
	camera->right = normalized(cross(camera->forward, z3f));
	camera->up = cross(camera->right, camera->forward);
}

static Ray
generate_camera_ray(Camera *camera, Image *image, i32 x, i32 y)
{
	Ray ray;

	float two_over_max_dim = 2.0f / max_val(image->width, image->height);
	float relative_x = (float)(x - 0.5f * image->width) * two_over_max_dim;
	float relative_y = (float)(y - 0.5f * image->height) * two_over_max_dim;

	ray.origin = camera->origin;

	ray.direction = camera->forward +
		relative_x * camera->tan_half_field_of_view * camera->right -
		relative_y * camera->tan_half_field_of_view * camera->up;

	ray.direction = normalized(ray.direction);

	return ray;
}

static Vec3f
evaluate_ray(Ray *ray, float t)
{
	return ray->origin + t * ray->direction;
}

#define epsilon 0.0001

static float
intersect_ray_plane(Ray *ray, Plane *plane)
{
	float numerator = dot(plane->origin - ray->origin, plane->normal);
	float denominator = dot(ray->direction, plane->normal);

	float t;
	if (fabs(denominator) < epsilon) {
		t = NAN;
	}
	else {
		t = numerator / denominator;
		if (t < epsilon) {
			t = NAN;
		}
	}

	return t;
}

static float
intersect_ray_sphere(Ray *ray, Sphere *sphere)
{
	Vec3f sphere_local_origin = ray->origin - sphere->center;
	float B = 2.0f * dot(sphere_local_origin, ray->direction);
	float C = squared_norm(sphere_local_origin) - sphere->radius * sphere->radius;

	float discriminant = B * B - 4 * C;

	if (discriminant <= epsilon) {
		return NAN;
	}

	float sqrt_discriminant = sqrtf(discriminant);

	float twice_t1 = -B - sqrt_discriminant;
	if (twice_t1 >= epsilon) {
		return 0.5f * twice_t1;
	}

	float twice_t2 = -B + sqrt_discriminant;
	if (twice_t2 >= epsilon) {
		return 0.5f * twice_t2;
	}

	return NAN;
}

dllExport PyObject *
PyInit_raytracer(void)
{
	return 0;
}

extern int
main(void)
{
	Image image[1];

	init_image(image, 1920, 1080);

	Scene scene;
	{ // Initialize the scene.
		scene.background_color = o3f;
		init_camera(&scene.camera, 40.0f / 180.0f * M_PI, vec3f(8.3f, -8.9f, 1.2f), vec3f(0.0f, 0.0f, 1.0f));

		static PointLight point_lights[1];
		point_lights[0].intensity = 1.0f;
		point_lights[0].position = vec3f(-3.5f, -2.0f, 3.0f);

		static Material materials[2];
		materials[0].color = vec3f(1.0f, 0.0f, 0.0f);
		materials[1].color = vec3f(0.0f, 1.0f, 0.0f);

		static Plane planes[1];
		planes[0].origin = o3f;
		planes[0].normal = z3f;
		planes[0].material_index = 0;

		static Sphere spheres[1];
		spheres[0].center = vec3f(0.0f, 0.0f, 1.0f);
		spheres[0].radius = 1.0f;
		spheres[0].material_index = 1;

		scene.materials = materials;
		scene.num_materials = arr_len(materials);

		scene.planes = planes;
		scene.num_planes = arr_len(planes);
		
		scene.spheres = spheres;
		scene.num_spheres = arr_len(spheres);

		scene.point_lights = point_lights;
		scene.num_point_lights = arr_len(point_lights);
	}

	for (i32 y = 0; y < image->height; y++) {
		for (i32 x = 0; x < image->width; x++) {
			u8 *pixel = get_image_pixel(image, x, y);

			Ray ray = generate_camera_ray(&scene.camera, image, x, y);

			bool intersected = false;

			float closest_t = INFINITY;
			i32 closest_material_index = -1;
			Vec3f closest_normal = o3f;

			for (i32 plane_index = 0; plane_index < scene.num_planes; plane_index++) {
				Plane *plane = scene.planes + plane_index;

				float t = intersect_ray_plane(&ray, plane);

				if (t < closest_t) {
					closest_t = t;
					closest_material_index = plane->material_index;
					closest_normal = plane->normal;
				}
			}


			for (i32 sphere_index = 0; sphere_index < scene.num_spheres; sphere_index++) {
				Sphere *sphere = scene.spheres + sphere_index;

				float t = intersect_ray_sphere(&ray, sphere);

				if (t < closest_t) {
					closest_t = t;
					closest_material_index = sphere->material_index;
					closest_normal = evaluate_ray(&ray, t) - sphere->center;
				}
			}

			pixel[3] = 255;
			
			// Do shading calculations.
			Vec3f pixelColor = scene.background_color;
			if (closest_material_index >= 0) {
				Material *material = scene.materials + closest_material_index;

				closest_normal = normalized(closest_normal);
				Vec3f hit_point = evaluate_ray(&ray, closest_t);

				pixelColor = o3f;
				for (i32 point_light_index = 0; point_light_index < scene.num_point_lights; point_light_index++) {
					PointLight *point_light = scene.point_lights + point_light_index;

					Vec3f c_cool = vec3f(0.0f, 0.0f, 0.3f) + 0.35 * material->color;
					Vec3f c_warm = vec3f(0.3f, 0.3f, 0.0f) + 0.35 * material->color;
					Vec3f c_highlight = vec3f(1.0f, 1.0f, 1.0f);

					Vec3f w_i = normalized(point_light->position - hit_point);
					Vec3f w_o = -ray.direction;

					float t = 0.5f * (dot(closest_normal, w_i) + 1.0f);
					Vec3f reflected = -w_i + 2.0f * dot(closest_normal, w_i) * closest_normal;
					float s = clamp01(100.0f * dot(reflected, w_o) - 97.0f);

					pixelColor += lerp(lerp(c_cool, c_warm, t), c_highlight, s);
				}
			}

			for (i32 i = 0; i < 3; i++) {
				pixel[i] = pixelColor.el[i] * 255;
			}
		}  
	}

	printf("Writing image to disk ...\n");
	stbi_write_png("out.png", image->width, image->height, 4, image->data, image->width * 4);

	terminate_image(image);

	return EXIT_SUCCESS;
}