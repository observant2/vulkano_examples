use gltf::accessor::Dimensions;
use gltf::json::accessor::ComponentType;
use gltf::Semantic;

pub struct Model {
    pub vertices: Vec<[f32; 3]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub indices: Vec<u16>,
}

impl Model {
    pub fn load(path: &str) -> Model {
        let (model, buffers, _) = gltf::import(path).unwrap_or_else(|_| panic!("couldn't load model at: {}", path));

        let mut vertices = Vec::<[f32; 3]>::new();
        let mut tex_coords = Vec::<[f32; 2]>::new();
        let mut indices = Vec::<u16>::new();

        for mesh in model.meshes() {
            for primitive in mesh.primitives() {
                for attr in primitive.attributes() {
                    if attr.0 == Semantic::Positions {
                        if attr.1.dimensions() != Dimensions::Vec3 {
                            panic!("Positions of gltf need to be vec3");
                        };

                        if attr.1.data_type() == ComponentType::F32 {
                            let accessor = attr.1;
                            let buffer_view = accessor.view().unwrap();
                            let contents: &[f32] = bytemuck::cast_slice(&buffers[0][buffer_view.offset()..(buffer_view.offset() + buffer_view.length())]);
                            for i in (0..accessor.count() * 3).step_by(3) {
                                vertices.push([contents[i], contents[i + 1], contents[i + 2], ]);
                            }
                        }
                    } else if attr.0 == Semantic::TexCoords(0) {
                        if attr.1.dimensions() != Dimensions::Vec2 {
                            panic!("TexCoords of gltf need to be vec2");
                        };

                        if attr.1.data_type() == ComponentType::F32 {
                            let accessor = attr.1;
                            let buffer_view = accessor.view().unwrap();
                            let contents: &[f32] = bytemuck::cast_slice(&buffers[0][buffer_view.offset()..(buffer_view.offset() + buffer_view.length())]);
                            for i in (0..accessor.count() * 2).step_by(2) {
                                tex_coords.push([contents[i], contents[i + 1]]);
                            }
                        }
                    }
                }

                if let Some(accessor) = primitive.indices() {
                    let buffer_view = accessor.view().unwrap();
                    let contents: &[u16] = bytemuck::cast_slice(&buffers[0][buffer_view.offset()..(buffer_view.offset() + buffer_view.length())]);
                    for val in contents {
                        indices.push(*val);
                    }
                }
            }
        }

        Model {
            vertices,
            indices,
            tex_coords,
        }
    }
}
