#pragma warning(disable:4996)

// Include this before any standard headers!
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define dllExport extern "C" __declspec(dllexport)
#define staticC extern "C" static

#ifndef M_PI
#define M_PI 3.141592653589793238
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// TODO: Roll own assert?
#define dbgAssert(x) assert(x)

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

union Vec4f {
	struct {
		float x, y, z, w;
	};

	struct {
		float r, g, b, a;
	};

	struct {
		Vec3f xyz;
		float dummy;
	};

	float el[4];
};

union Mat4f {
	Vec4f rows[4];
	float el[4][4];
};

static void
print_mat4f(Mat4f matrix)
{
	printf("(%.4f %.4f %.4f %.4f\n",  matrix.el[0][0], matrix.el[0][1], matrix.el[0][2], matrix.el[0][3]);
	printf(" %.4f %.4f %.4f %.4f\n",  matrix.el[1][0], matrix.el[1][1], matrix.el[1][2], matrix.el[1][3]);
	printf(" %.4f %.4f %.4f %.4f\n",  matrix.el[2][0], matrix.el[2][1], matrix.el[2][2], matrix.el[2][3]);
	printf(" %.4f %.4f %.4f %.4f)\n", matrix.el[3][0], matrix.el[3][1], matrix.el[3][2], matrix.el[3][3]);
}


static Vec3f
vec3f(float x, float y, float z)
{
	Vec3f v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}

static Vec4f
vec4f(float x, float y, float z, float w)
{
	Vec4f v;
	v.x = x;
	v.y = y;
	v.z = z;
	v.w = w;
	return v;
}

static Vec4f
vec4f(Vec3f xyz, float w)
{
	Vec4f v;
	v.xyz = xyz;
	v.w = w;
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

static Vec3f
perspective_divide(Vec4f v)
{
	return vec3f(v.x / v.w, v.y / v.w, v.z / v.w);
}

static float
dot(Vec4f a, Vec4f b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
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

static Vec4f
operator * (Mat4f m, Vec4f v)
{
	Vec4f result;
	result.x = dot(m.rows[0], v);
	result.y = dot(m.rows[1], v);
	result.z = dot(m.rows[2], v);
	result.w = dot(m.rows[3], v);
	return result;
}

static Mat4f
operator / (Mat4f m, float f)
{
	for (i32 i = 0; i < 4; i++) {
		for (i32 j = 0; j < 4; j++) {
			m.el[i][j] /= f;
		}
	}

	return m;
}

static Mat4f
inv_mat4f(Mat4f m)
{
	Mat4f result;

	// Compute the adjugate matrix
#define sarrus(a11, a12, a13, a21, a22, a23, a31, a32, a33) \
	(a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a11*a23*a32 - a12*a21*a33 - a13*a22*a31)

#define cofactor(ro1, ro2, ro3, col1, col2, col3) \
	sarrus( \
		m.el[ro1][col1], m.el[ro1][col2], m.el[ro1][col3], \
		m.el[ro2][col1], m.el[ro2][col2], m.el[ro2][col3], \
		m.el[ro3][col1], m.el[ro3][col2], m.el[ro3][col3] \
	)

	result.el[0][0] = +cofactor(1, 2, 3, 1, 2, 3);
	result.el[0][1] = -cofactor(0, 2, 3, 1, 2, 3);
	result.el[0][2] = +cofactor(0, 1, 3, 1, 2, 3);
	result.el[0][3] = -cofactor(0, 1, 2, 1, 2, 3);
	result.el[1][0] = -cofactor(1, 2, 3, 0, 2, 3);
	result.el[1][1] = +cofactor(0, 2, 3, 0, 2, 3);
	result.el[1][2] = -cofactor(0, 1, 3, 0, 2, 3);
	result.el[1][3] = +cofactor(0, 1, 2, 0, 2, 3);
	result.el[2][0] = +cofactor(1, 2, 3, 0, 1, 3);
	result.el[2][1] = -cofactor(0, 2, 3, 0, 1, 3);
	result.el[2][2] = +cofactor(0, 1, 3, 0, 1, 3);
	result.el[2][3] = -cofactor(0, 1, 2, 0, 1, 3);
	result.el[3][0] = -cofactor(1, 2, 3, 0, 1, 2);
	result.el[3][1] = +cofactor(0, 2, 3, 0, 1, 2);
	result.el[3][2] = -cofactor(0, 1, 3, 0, 1, 2);
	result.el[3][3] = +cofactor(0, 1, 2, 0, 1, 2);

#undef cofactor
#undef sarrus

	float det = m.el[0][0]*result.el[0][0] + m.el[1][0]*result.el[0][1] + m.el[2][0]*result.el[0][2] + m.el[3][0] * result.el[0][3];
	
	return result / det;
}

// RGBA 8 bits per channel.
enum ImageFormat {
	ImageFormat_rgba8,
	ImageFormat_rgba32f
};

struct Image {
	bool freeDataOnTermination;
	ImageFormat format;

	i32 width;
	i32 height;

	union {
		void *data_opaque;
		u8 *data_u8;
		float *data_float;
	};
};

struct Camera {
	Mat4f inv_view_proj_matrix;
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
	Mat4f model_matrix;
	Mat4f inv_model_matrix;

	i32 material_index;
};

struct PointLight {
	Vec3f position;
	float intensity;
};

struct Scene {
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
init_image(Image *image, ImageFormat format, i32 width, i32 height, void *data = 0)
{
	image->format = format;

	image->width = width;
	image->height = height;

	image->freeDataOnTermination = true;
	if (data) {
		image->freeDataOnTermination = true;
		image->data_opaque = data;
	} else if (format == ImageFormat_rgba32f) {
		image->data_float = (float *)malloc(width * height * 32);
	} else if (format == ImageFormat_rgba8) {
		image->data_u8 = (u8 *)malloc(width * height * 4);
	}
}

static void
terminate_image(Image *image)
{
	if (image->freeDataOnTermination) {
		free(image->data_opaque);
	}
}

static void
write_image_pixel(Image *image, i32 x, i32 y, Vec4f color)
{
	switch (image->format) {
	case ImageFormat_rgba32f: {
		float *px = image->data_float + 4 * (y * image->width + x);
		for (i32 i = 0; i < 4; i++) {
			px[i] = color.el[i];
		}
	} break;

	case ImageFormat_rgba8: {
		u8 *px = image->data_u8 + 4 * (y * image->width + x);
		for (i32 i = 0; i < 4; i++) {
			px[i] = color.el[i] * 255;
		}
	} break;
	}
}

static void
init_camera(Camera *camera, float field_of_view, Vec3f origin, Vec3f look_at)
{
	// TODO: Compute view projection matrix?
	dbgAssert(0);
}

static Ray
generate_camera_ray(Camera *camera, Image *image, i32 x, i32 y)
{
	Ray ray;

	float two_over_max_dim = 2.0f / max_val(image->width, image->height);
	float relative_x = (float)(x - 0.5f * image->width) * two_over_max_dim;
	float relative_y = (float)(y - 0.5f * image->height) * two_over_max_dim;

	ray.origin = perspective_divide(
		camera->inv_view_proj_matrix * 
		vec4f(2.0f * x / image->width - 1.0f, 2.0f * y / image->height - 1.0f, -1.0f, 1.0f)
	);

	Vec3f target = perspective_divide(
		camera->inv_view_proj_matrix *
		vec4f(2.0f * x / image->width - 1.0f, 2.0f * y / image->height - 1.0f, 0.0f, 1.0f)
	);
	ray.direction = normalized(target - ray.origin);

	return ray;
}

static Vec3f
evaluate_ray(Ray *ray, float t)
{
	return ray->origin + t * ray->direction;
}

static Ray
operator * (Mat4f m, Ray ray)
{
	Ray result;
	result.origin = (m * vec4f(ray.origin, 1.0f)).xyz;
	result.direction = (m * vec4f(ray.direction, 0.0f)).xyz;
	return result;
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
	Ray local_ray = sphere->inv_model_matrix * (*ray);
	float local_ray_len = norm(local_ray.direction);
	local_ray.direction *= 1.0f / local_ray_len;

	float B = 2.0f * dot(local_ray.origin, local_ray.direction);
	float C = squared_norm(local_ray.origin) - 1.0f;

	float discriminant = B * B - 4 * C;

	if (discriminant <= epsilon) {
		return NAN;
	}

	float sqrt_discriminant = sqrtf(discriminant);

	float twice_t1 = -B - sqrt_discriminant;
	if (twice_t1 >= epsilon) {
		return 0.5f * twice_t1 / local_ray_len;
	}

	float twice_t2 = -B + sqrt_discriminant;
	if (twice_t2 >= epsilon) {
		return 0.5f * twice_t2 / local_ray_len;
	}

	return NAN;
}

struct PythonImage {
	PyObject_HEAD
	Image image;
};

static void
register_python_image_type()
{
	static PyTypeObject python_image_type ={
		PyVarObject_HEAD_INIT(NULL, 0)
	};
	python_image_type.tp_name = "tutrt.PythonImage";
	python_image_type.tp_basicsize = sizeof(PythonImage);
	python_image_type.tp_itemsize = 0;
	python_image_type.tp_flags = Py_TPFLAGS_DEFAULT;
	python_image_type.tp_new = PyType_GenericNew;
}


static void
render_scene(Image *image, Scene *scene, Camera *camera)
{
	for (i32 y = 0; y < image->height; y++) {
		for (i32 x = 0; x < image->width; x++) {
			Ray ray = generate_camera_ray(camera, image, x, y);

			bool intersected = false;

			float closest_t = INFINITY;
			i32 closest_material_index = -1;
			Vec3f closest_normal = o3f;

			for (i32 plane_index = 0; plane_index < scene->num_planes; plane_index++) {
				Plane *plane = scene->planes + plane_index;

				float t = intersect_ray_plane(&ray, plane);

				if (t < closest_t) {
					closest_t = t;
					closest_material_index = plane->material_index;
					closest_normal = plane->normal;
				}
			}


			for (i32 sphere_index = 0; sphere_index < scene->num_spheres; sphere_index++) {
				Sphere *sphere = scene->spheres + sphere_index;

				float t = intersect_ray_sphere(&ray, sphere);

				if (t < closest_t) {
					closest_t = t;
					closest_material_index = sphere->material_index;
					closest_normal = evaluate_ray(&ray, t) - vec3f(sphere->model_matrix.el[0][0], sphere->model_matrix.el[1][0], sphere->model_matrix.el[2][0]);
				}
			}

			// Do shading calculations.
			Vec3f pixelColor = scene->background_color;
			if (closest_material_index >= 0) {
				Material *material = scene->materials + closest_material_index;

				closest_normal = normalized(closest_normal);
				Vec3f hit_point = evaluate_ray(&ray, closest_t);

				pixelColor = o3f;
				for (i32 point_light_index = 0; point_light_index < scene->num_point_lights; point_light_index++) {
					PointLight *point_light = scene->point_lights + point_light_index;

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

			write_image_pixel(image, x, y, vec4f(pixelColor, 1.0f));
		}
	}
}

static int
python_read_mat4f(PyObject *python_matrix, Mat4f *r_result)
{
	Mat4f result;
	{
		PyObject *row_iter = PyObject_GetIter(python_matrix);
		if (!row_iter) return -1;

		PyObject *row;
		i32 i = 0;
		while (row = PyIter_Next(row_iter)) {
			PyObject *entry_iter = PyObject_GetIter(row);
			if (!entry_iter) return -1;

			PyObject *entry;
			i32 j = 0;
			while (entry = PyIter_Next(entry_iter)) {
				result.el[i][j] = PyFloat_AsDouble(entry);
				Py_DECREF(entry);
				j++;
			}

			Py_DECREF(entry_iter);
			Py_DECREF(row);
			i++;
		}
		Py_DECREF(row_iter);
	}

	*r_result = result;

	return 0;
}

staticC PyObject *
python_render(PyObject *self, PyObject *args)
{
	PyObject *python_image;
	PyObject *python_camera;
	PyObject *python_scene;

	if (!PyArg_ParseTuple(args, "OOO", &python_image, &python_camera, &python_scene)) {
		return 0;
	}

	PyObject *dimensions = PyObject_GetAttrString(python_image, "dimensions");
	if (!dimensions) return 0;
	printf("dimensions %p\n", dimensions);

	PyObject *python_width = PyTuple_GetItem(dimensions, 0);
	if (!python_width) return 0;
	i32 width = PyLong_AsLong(python_width);

	PyObject *python_height = PyTuple_GetItem(dimensions, 1);
	if (!python_height) return 0;
	i32 height = PyLong_AsLong(python_height);

	PyObject *python_buffer = PyObject_GetAttrString(python_image, "buffer");
	if (!python_buffer) return 0;

	PyObject *python_inv_view_proj_matrix = PyObject_GetAttrString(python_camera, "inv_view_proj_matrix");
	if (!python_inv_view_proj_matrix) return 0;

	Camera camera;
	if (python_read_mat4f(python_inv_view_proj_matrix, &camera.inv_view_proj_matrix) < 0) return 0;
	Py_DECREF(python_inv_view_proj_matrix);

	Image image;

	printf("dimensions %ldx%ld\n", width, height);

	Scene scene;

	Py_buffer buffer_view = {};
	if (PyObject_GetBuffer(python_buffer, &buffer_view, PyBUF_WRITABLE) < 0) {
		return 0;
	}

	void *buffer = buffer_view.buf;
	printf("Buffer: %p\n", buffer);
	init_image(&image, ImageFormat_rgba32f, width, height, buffer);

	//init_camera(&camera, 40.0f / 180.0f * M_PI, vec3f(8.3f, -8.9f, 1.2f), vec3f(0.0f, 0.0f, 1.0f));

	scene.background_color = o3f;

	static PointLight point_lights[1];
	point_lights[0].intensity = 1.0f;
	point_lights[0].position = vec3f(-3.5f, -2.0f, 3.0f);
	scene.point_lights = point_lights;
	scene.num_point_lights = arr_len(point_lights);

	static Material materials[2];
	materials[0].color = vec3f(1.0f, 0.0f, 0.0f);
	materials[1].color = vec3f(0.0f, 1.0f, 0.0f);
	scene.materials = materials;
	scene.num_materials = arr_len(materials);

	static Plane planes[1];
	planes[0].origin = o3f;
	planes[0].normal = z3f;
	planes[0].material_index = 0;
	scene.planes = planes;
	scene.num_planes = arr_len(planes);

	PyObject *python_spheres = PyObject_GetAttrString(python_scene, "spheres");
	scene.num_spheres = PyObject_Length(python_spheres);
	printf("num_spheres is %d\n", scene.num_spheres);
	scene.spheres = (Sphere *)malloc(sizeof(Sphere) * scene.num_spheres);

	PyObject *sphere_iter = PyObject_GetIter(python_spheres);
	if (!sphere_iter) return 0;
	PyObject *python_sphere;
	Sphere *sphere = scene.spheres;
	while (python_sphere = PyIter_Next(sphere_iter)) {
		printf("Adding sphere %p\n", sphere);

		PyObject *python_model_matrix = PyObject_GetAttrString(python_sphere, "model_matrix");
		if (!python_model_matrix) return 0;
		if (python_read_mat4f(python_model_matrix, &sphere->model_matrix) < 0) return 0;
		Py_DECREF(python_model_matrix);

		sphere->inv_model_matrix = inv_mat4f(sphere->model_matrix);
		print_mat4f(sphere->inv_model_matrix);

		sphere->material_index = 1;


		Py_DECREF(python_sphere);
		sphere++;
	}
	Py_DECREF(sphere_iter);
	Py_DECREF(python_spheres);

	printf("Rendering scene");
	render_scene(&image, &scene, &camera);

	free(scene.spheres);

	PyBuffer_Release(&buffer_view);
	Py_DECREF(python_buffer);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyMethodDef python_functions[] ={
	{"render", python_render, METH_VARARGS, "Render an image."},
	{nullptr, nullptr, 0, nullptr}
};

static PyModuleDef python_module ={
	PyModuleDef_HEAD_INIT,
	"raytracer",
	nullptr,
	-1,
	python_functions
};

dllExport PyObject *
PyInit_raytracer(void)
{
	return PyModule_Create(&python_module);
}
