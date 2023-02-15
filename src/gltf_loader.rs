use std::ops::Mul;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use gltf::accessor::Dimensions;
use gltf::json::accessor::ComponentType;
use gltf::{Node, Semantic};
use gltf::buffer::Data;
use nalgebra_glm::{Mat4, mat4_to_mat3, vec3, vec4};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::buffer::{Buffer, BufferAllocateInfo, BufferUsage, Subbuffer};
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod, Vertex)]
pub struct GltfVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

pub struct Scene {
    // root nodes
    pub nodes: Vec<ModelNode>,
    pub vertices: Vec<GltfVertex>,
    pub indices: Vec<u16>,
    pub vertex_buffer: Subbuffer<[GltfVertex]>,
    pub index_buffer: Subbuffer<[u16]>,
}

struct SceneBuilder {
    pub nodes: Vec<ModelNode>,
    pub vertices: Vec<GltfVertex>,
    pub indices: Vec<u16>,
    pub vertex_buffer: Option<Subbuffer<[GltfVertex]>>,
    pub index_buffer: Option<Subbuffer<[u16]>>,
}

impl SceneBuilder {
    pub fn build(self) -> Scene {
        Scene {
            nodes: self.nodes,
            vertices: self.vertices,
            indices: self.indices,
            vertex_buffer: self.vertex_buffer.expect("gltf_loader: vertex_buffer not initialized!"),
            index_buffer: self.index_buffer.expect("gltf_loader: vertex_buffer not initialized!"),
        }
    }

    fn add_mesh_data(&mut self, mesh: &gltf::Mesh, buffers: &[Data]) -> Mesh {
        let mut primitives = vec![];

        for primitive in mesh.primitives() {
            let mut vertices = Vec::<[f32; 3]>::new();
            let mut normals = Vec::<[f32; 3]>::new();
            let mut tex_coords = Vec::<[f32; 2]>::new();
            let mut colors = Vec::<[f32; 4]>::new();
            let mut indices = Vec::<u16>::new();

            let first_index = self.indices.len();
            let first_vertex = self.vertices.len() as u16;
            let mut index_count = 0;
            let mut vertex_count = 0;

            for attr in primitive.attributes() {
                if attr.0 == Semantic::Positions {
                    if attr.1.dimensions() != Dimensions::Vec3 {
                        panic!("Positions of gltf need to be vec3");
                    };

                    if attr.1.data_type() == ComponentType::F32 {
                        let accessor = attr.1;
                        let buffer_view = accessor.view().unwrap();
                        let contents: &[f32] = bytemuck::cast_slice(&buffers[0][buffer_view.offset()..][..buffer_view.length()]);
                        for i in (0..accessor.count() * 3).step_by(3) {
                            vertices.push([contents[i], contents[i + 1], contents[i + 2]]);
                        }
                        vertex_count += accessor.count();
                    }
                } else if attr.0 == Semantic::Normals
                {
                    if attr.1.dimensions() != Dimensions::Vec3 {
                        panic!("Normals of gltf need to be vec3");
                    };

                    if attr.1.data_type() == ComponentType::F32 {
                        let accessor = attr.1;
                        let buffer_view = accessor.view().unwrap();
                        let contents: &[f32] = bytemuck::cast_slice(&buffers[0][buffer_view.offset()..][..buffer_view.length()]);
                        for i in (0..accessor.count() * 3).step_by(3) {
                            let normal = vec3(contents[i], contents[i + 1], contents[i + 2]).normalize();
                            normals.push(normal.into());
                        }
                    }
                } else if attr.0 == Semantic::TexCoords(0) {
                    if attr.1.dimensions() != Dimensions::Vec2 {
                        panic!("TexCoords of gltf need to be vec2");
                    };

                    if attr.1.data_type() == ComponentType::F32 {
                        let accessor = attr.1;
                        let buffer_view = accessor.view().unwrap();
                        let contents: &[f32] = bytemuck::cast_slice(&buffers[0][buffer_view.offset()..][..buffer_view.length()]);
                        for i in (0..accessor.count() * 2).step_by(2) {
                            tex_coords.push([contents[i], contents[i + 1]]);
                        }
                    }
                }
            }

            if let Some(accessor) = primitive.indices() {
                let buffer_view = accessor.view().unwrap();
                // TODO: Take into account other than unsigned short!
                let contents: &[u16] = bytemuck::cast_slice(&buffers[0][buffer_view.offset()..][..buffer_view.length()]);
                for val in contents {
                    indices.push(*val + first_vertex);
                }
                index_count += accessor.count();
            }

            colors.push(primitive.material().pbr_metallic_roughness().base_color_factor());

            if tex_coords.len() < vertices.len() {
                for _ in 0..vertices.len() {
                    tex_coords.push([0.0, 0.0]);
                }
            }

            if colors.len() < vertices.len() {
                colors.resize(vertices.len(), colors[0]);
            }

            for (v, (n, (uv, c))) in vertices.iter().zip(normals.iter().zip(tex_coords.iter().zip(colors.iter()))) {
                self.vertices.push(GltfVertex {
                    position: *v,
                    normal: *n,
                    color: *c,
                    uv: *uv,
                })
            }

            self.indices.extend(indices);

            let model_primitive = ModelPrimitive {
                first_index,
                index_count,
                first_vertex: first_vertex as usize,
                vertex_count,
                // TODO
                material_index: 0,
            };
            primitives.push(model_primitive);
        }

        Mesh {
            primitives,
        }
    }

    fn load_node(&mut self, node: &Node, parent: Option<&mut ModelNode>, buffers: &[Data]) {
        let transform = Mat4::from(node.transform().matrix());

        let mesh = if let Some(mesh) = node.mesh() {
            self.add_mesh_data(&mesh, buffers)
        } else {
            Mesh {
                primitives: vec![],
            }
        };

        let mut model_node = ModelNode {
            id: self.nodes.len(),
            transform,
            mesh,
            parent: None,
            children: vec![],
        };

        for child in node.children() {
            self.load_node(&child, Some(&mut model_node), buffers);
        }

        if let Some(parent) = parent {
            model_node.parent = Some(parent.id);
            parent.children.push(model_node.id);
        }
        self.nodes.push(model_node);
    }
}

pub struct Mesh {
    pub primitives: Vec<ModelPrimitive>,
}

pub type ModelNodeId = usize;

pub struct ModelNode {
    pub id: ModelNodeId,
    pub parent: Option<ModelNodeId>,
    pub children: Vec<ModelNodeId>,
    pub mesh: Mesh,
    pub transform: Mat4,
}

/// provides indices into the scene's vertex/index vector
pub struct ModelPrimitive {
    pub first_index: usize,
    pub index_count: usize,
    pub first_vertex: usize,
    pub vertex_count: usize,
    pub material_index: usize,
}

impl Scene {
    pub fn load(path: &str, memory_allocator: &Arc<StandardMemoryAllocator>, flip_y: bool, apply_transforms: bool) -> Scene {
        let m = gltf::import(path);

        let (model, buffers, _) = m.unwrap();

        let scene = &model.scenes().collect::<Vec<_>>()[0];

        let mut model = SceneBuilder {
            indices: vec![],
            vertices: vec![],
            nodes: vec![],
            index_buffer: None,
            vertex_buffer: None,
        };

        for node in scene.nodes() {
            model.load_node(&node, None, &buffers);
        }

        if apply_transforms {
            for node in &model.nodes {
                for primitive in &node.mesh.primitives {
                    for v in &mut model.vertices[primitive.first_vertex..][..primitive.vertex_count] {
                        let vertex = vec4(v.position[0], v.position[1], v.position[2], 1.0);
                        let res = node.transform.mul(&vertex);
                        v.position = [res[0], if flip_y { -1.0 } else { 1.0 } * res[1], res[2]];

                        let normal_matrix = mat4_to_mat3(&node.transform).try_inverse().unwrap().transpose();
                        let res = normal_matrix.mul(&vec3(v.normal[0], v.normal[1], v.normal[2]));

                        v.normal = [res[0], if flip_y { -1.0 } else { 1.0 } * res[1], res[2]];
                    }
                }
            }
        }

        model.vertex_buffer = Some(Buffer::from_iter(
            memory_allocator.as_ref(),
            BufferAllocateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                ..BufferAllocateInfo::default()
            },
            model.vertices.clone(),
        ).expect("failed to create vertex buffer"));

        model.index_buffer = Some(Buffer::from_iter(
            memory_allocator.as_ref(),
            BufferAllocateInfo {
                buffer_usage: BufferUsage::INDEX_BUFFER,
                ..BufferAllocateInfo::default()
            },
            model.indices.clone(),
        ).expect("failed to create index buffer"));

        model.build()
    }

    fn draw_node(&self, node: &ModelNode, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        for primitive in &node.mesh.primitives {
            builder.draw_indexed(primitive.index_count as u32, 1, primitive.first_index as u32, 0, 0).unwrap();
        }

        for child_id in &node.children {
            let child = &self.nodes[*child_id];
            self.draw_node(child, builder);
        }
    }

    pub fn draw(&self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        builder.bind_vertex_buffers(0, self.vertex_buffer.clone());
        builder.bind_index_buffer(self.index_buffer.clone());

        for node in &self.nodes {
            self.draw_node(node, builder);
        }
    }
}
