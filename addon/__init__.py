# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "TutRaytracer",
    "author" : "Kompilierbar",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 1),
    "location" : "",
    "warning" : "",
    "category" : "Generic"
}

import bpy
import array
import gpu
from gpu_extras.presets import draw_texture_2d
from mathutils import *

import addon.bin.raytracer as rt

class Image:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        width, height = dimensions
        #self.buffer = bytearray(width * height * 4)
        self.buffer = gpu.types.Buffer("FLOAT", width * height * 4)

class PointLight:
    def __init__(self, position, strength):
        self.position = position
        self.strength = strength

class Sphere:
    def __init__(self, model_matrix):
        self.model_matrix = model_matrix

class Plane:
    def __init__(self, model_matrix):
        self.model_matrix = model_matrix

class Scene:
    def __init__(self):
        self.spheres = []
        self.planes = []
        self.point_lights = []

class Camera:
    def __init__(self, view_projection_matrix):
        self.inv_view_proj_matrix = view_projection_matrix.inverted()

class TutRTEngine(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "TUTRT"
    bl_label = "TutRT"
    bl_use_preview = True

    # Init is called whenever a new render engine instance is created. Multiple
    # instances may exist at the same time, for example for a viewport and final
    # render.
    def __init__(self):
        self.scene = None
        self.camera = None

    # When the render engine instance is destroyed, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        pass

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.
    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)

        # Fill the render result with a flat color. The framebuffer is
        # defined as a list of pixels, each pixel itself being a list of
        # R,G,B,A values.
        if self.is_preview:
            color = [0.1, 0.2, 0.1, 1.0]
        else:
            color = [0.2, 0.1, 0.1, 1.0]

        pixel_count = self.size_x * self.size_y
        rect = [color] * pixel_count

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]
        layer.rect = rect
        self.end_result(result)

    # For viewport renders, this method gets called once at the start and
    # whenever the scene or 3D viewport changes. This method is where data
    # should be read from Blender in the same thread. Typically a render
    # thread will be started to do the work while keeping Blender responsive.
    def view_update(self, context, depsgraph):
        region = context.region
        view3d = context.space_data
        scene = depsgraph.scene

        # Get viewport dimensions
        dimensions = region.width, region.height

        print("View update running")

        if not self.scene:
            # First time initialization
            self.scene = Scene()
            first_time = True

            # Loop over all datablocks used in the scene.
            for datablock in depsgraph.ids:
                pass
        else:
            first_time = False

            # Test which datablocks changed
            for update in depsgraph.updates:
                print("Datablock updated: ", update.id.name)

            # Test if any material was added, removed or changed.
            if depsgraph.id_type_updated('MATERIAL'):
                print("Materials updated")

        # Loop over all object instances in the scene.
        if first_time or depsgraph.id_type_updated('OBJECT'):
            for instance in depsgraph.object_instances:
                pass

    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def view_draw(self, context, depsgraph):
        print("view_draw called")
        region = context.region
        scene = depsgraph.scene

        # Get viewport dimensions
        dimensions = region.width, region.height
        view_projection_matrix = context.region_data.perspective_matrix# * context.region_data.view_matrix.inverted()

        # Bind shader that converts from scene linear to display space,
        gpu.state.blend_set('ALPHA_PREMULT')
        self.bind_display_space_shader(scene)

        self.camera = Camera(view_projection_matrix)

        # Generate dummy float image buffer
        width, height = dimensions

        image = Image(dimensions)
        print(self.camera.inv_view_proj_matrix.inverted())

        self.scene = Scene()

        for instance in depsgraph.object_instances:
            object = instance.object.original
            tutrt_type = object.get("tutrt_type", "None")

            if object.type == "LIGHT":
                self.scene.point_lights.append(PointLight(object.location.copy(), object.data.energy * object.data.color))
            elif tutrt_type == "SPHERE":
                self.scene.spheres.append(Sphere(instance.matrix_world.copy()))
            elif tutrt_type == "PLANE":
                self.scene.planes.append(Plane(instance.matrix_world.copy()))

        rt.render(image, self.camera, self.scene)
        #print(image.buffer)

        # Generate texture

        #buffer = gpu.types.Buffer("UBYTE", width * height * 4, image.buffer)
        self.texture = gpu.types.GPUTexture((width, height), format="RGBA8", data=image.buffer)
        print(image.buffer[0:4])
        
        #draw_texture_2d(self.texture, (0, self.texture.height), self.texture.width, -self.texture.height)
        draw_texture_2d(self.texture, (0, 0), self.texture.width, self.texture.height)

        self.unbind_display_space_shader()
        gpu.state.blend_set('NONE')

class CustomDrawData:
    def __init__(self, dimensions):
        # Generate dummy float image buffer
        self.dimensions = dimensions
        width, height = dimensions

        pixels = width * height * array.array('f', [0.1, 0.2, 0.1, 1.0])
        pixels = gpu.types.Buffer('FLOAT', width * height * 4, pixels)

        # Generate texture
        self.texture = gpu.types.GPUTexture((width, height), format='RGBA16F', data=pixels)

        # Note: This is just a didactic example.
        # In this case it would be more convenient to fill the texture with:
        # self.texture.clear('FLOAT', value=[0.1, 0.2, 0.1, 1.0])

    def __del__(self):
        del self.texture

    def draw(self):
        draw_texture_2d(self.texture, (0, self.texture.height), self.texture.width, -self.texture.height)

class OBJECT_OT_add(bpy.types.Operator):
    bl_idname = "mesh.add_tutrt_object"
    bl_label = "Add TutRT object"
    bl_options = {"REGISTER", "UNDO"}
    type: bpy.props.StringProperty(name="type", default="sphere")

    def execute(self, context):
        if self.type == "SPHERE":
            bpy.ops.mesh.primitive_uv_sphere_add()
        elif self.type == "PLANE":
            bpy.ops.mesh.primitive_plane_add()

        context.object["tutrt_type"] = self.type

        return {"FINISHED"}

class VIEW3D_MT_tutrt_add(bpy.types.Menu):
    bl_label = "TutRT"
    bl_idname = "VIEW3D_MT_tutrt_add"

    def draw(self, context):
        layout = self.layout

        layout.operator(OBJECT_OT_add.bl_idname, text="Sphere").type = "SPHERE"
        layout.operator(OBJECT_OT_add.bl_idname, text="Plane").type = "PLANE"

def add_objects_submenu(self, context):
    self.layout.menu(VIEW3D_MT_tutrt_add.bl_idname)

class DATA_PT_light(bpy.types.Panel):
    bl_label = "Light"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return context.light and context.engine == TutRTEngine.bl_idname

    def draw(self, context):
        layout = self.layout

        light = context.light

        layout.row().prop(light, "type")
        layout.row().prop(light, "energy")
        layout.row().prop(light, "color")

def register():
    bpy.utils.register_class(TutRTEngine)
    bpy.utils.register_class(OBJECT_OT_add)
    bpy.utils.register_class(VIEW3D_MT_tutrt_add)
    bpy.utils.register_class(DATA_PT_light)

    bpy.types.VIEW3D_MT_add.prepend(add_objects_submenu)


def unregister():
    bpy.utils.unregister_class(DATA_PT_light)
    bpy.utils.unregister_class(VIEW3D_MT_tutrt_add)
    bpy.utils.unregister_class(OBJECT_OT_add)
    bpy.utils.unregister_class(TutRTEngine)

    bpy.types.VIEW3D_MT_add.remove(add_objects_submenu)