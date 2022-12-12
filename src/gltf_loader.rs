use gltf::accessor::Dimensions;
use gltf::json::accessor::ComponentType;
use gltf::scene::Transform;
use gltf::Semantic;
use nalgebra_glm::{Mat4, vec4, Vec4};

pub struct Model {
    pub meshes: Vec<Mesh>,
}

pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub indices: Vec<u16>,
    pub transform: Option<Mat4>,
    pub colors: Vec<[f32; 4]>,
}

impl Model {
    pub fn load(path: &str) -> Model {
        let (model, buffers, _) = gltf::import(path).unwrap_or_else(|_| panic!("couldn't load model at: {}", path));

        let mut meshes: Vec<Mesh> = vec![];

        for mesh in model.meshes() {
            let mut vertices = Vec::<[f32; 3]>::new();
            let mut normals = Vec::<[f32; 3]>::new();
            let mut tex_coords = Vec::<[f32; 2]>::new();
            let mut colors = Vec::<[f32; 4]>::new();
            let mut indices = Vec::<u16>::new();

            for primitive in mesh.primitives() {
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
                                vertices.push([contents[i], contents[i + 1], contents[i + 2], ]);
                            }
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
                                normals.push([contents[i], contents[i + 1], contents[i + 2], ]);
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
                    let contents: &[u16] = bytemuck::cast_slice(&buffers[0][buffer_view.offset()..][..buffer_view.length()]);
                    for val in contents {
                        indices.push(*val);
                    }
                }

                colors.push(primitive.material().pbr_metallic_roughness().base_color_factor());
            }

            if tex_coords.len() < vertices.len() {
                for _ in 0..vertices.len() {
                    tex_coords.push([0.0, 0.0]);
                }
            }

            if colors.len() < vertices.len() {
                colors.resize(vertices.len(), colors[0]);
            }

            meshes.push(Mesh {
                tex_coords,
                indices,
                vertices,
                normals,
                colors,
                transform: None,
            })
        }

        for n in model.nodes() {
            if let Some(mesh) = n.mesh() {
                // node applies to mesh
                for v in &mut meshes[mesh.index()].vertices {
                    let res: Vec4 = (Mat4::from(n.transform().matrix()) * vec4(v[0], v[1], v[2], 1.0));
                    // bruuuh
                    let arr = res.data.0[0];
                    *v = [arr[0], arr[1], arr[2]];
                }
            }
        }

        for n in model.nodes() {
            for c in n.children() {
                if let Some(mesh) = c.mesh() {
                    for v in &mut meshes[mesh.index()].vertices {
                        let res: Vec4 = (Mat4::from(n.transform().matrix()) * vec4(v[0], v[1], v[2], 1.0));
                        // bruuuh
                        let arr = res.data.0[0];
                        *v = [arr[0], arr[1], arr[2]];
                    }
                }
            }
        }

        Model {
            meshes,
        }
    }
}
