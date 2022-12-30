use nalgebra_glm::{Mat4, rotate_x, rotate_y, rotate_z, translate, Vec3, vec3};
use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, VirtualKeyCode};
use winit::event::DeviceEvent::{Key, MouseMotion};
use winit::event::WindowEvent::{MouseInput, MouseWheel};

pub struct Camera {
    view_matrix: Mat4,
    perspective_matrix: Mat4,
    keys_pressed: KeysPressed,
    pub movement_speed: f32,
    position: Vec3,
    rotation: Vec3,
    pub mouse_pressed: (MouseButton, bool),
    pub camera_type: CameraType,
    pub updated: bool,
}

#[derive(PartialEq)]
pub enum CameraType {
    LookAt,
    FirstPerson,
}

#[derive(Default)]
pub struct KeysPressed {
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

impl Camera {
    pub fn new(position: Vec3, aspect: f32, fovy: f32, near: f32, far: f32) -> Self {
        let mut camera = Camera {
            position,
            rotation: vec3(0.0, 0.0, 0.0),
            view_matrix: Mat4::identity(),
            perspective_matrix: nalgebra_glm::perspective(aspect, fovy, near, far),
            mouse_pressed: (MouseButton::Left, false),
            camera_type: CameraType::LookAt,
            keys_pressed: KeysPressed::default(),
            movement_speed: 3.0,
            updated: false,
        };
        camera.update_view_matrix();

        camera
    }

    pub fn handle_input(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent {
                event: MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_x, y),
                    ..
                },
                ..
            } => {
                self.translate(vec3(0.0, 0.0, y * 2.0));
            }
            Event::DeviceEvent {
                event: MouseMotion {
                    delta: (x, y), ..
                },
                ..
            } => {
                let scale = 1.0;

                if self.mouse_pressed.1 {
                    self.set_rotation([self.rotation.x - *y as f32 / scale,
                        self.rotation.y + *x as f32 / scale,
                        self.rotation.z].into());
                }
            }
            Event::WindowEvent {
                event: MouseInput {
                    state,
                    ..
                },
                ..
            } => {
                self.mouse_pressed.1 = *state == ElementState::Pressed;
            }
            Event::DeviceEvent {
                event: Key {
                    0: key_input
                },
                ..
            } => {
                let pressed = key_input.state == ElementState::Pressed;
                match key_input.virtual_keycode {
                    Some(VirtualKeyCode::W) => {
                        self.keys_pressed.up = pressed;
                        self.update_view_matrix();
                        self.updated = false;
                    }
                    Some(VirtualKeyCode::S) => {
                        self.keys_pressed.down = pressed;
                        self.update_view_matrix();
                        self.updated = false;
                    }
                    Some(VirtualKeyCode::A) => {
                        self.keys_pressed.left = pressed;
                        self.update_view_matrix();
                        self.updated = false;
                    }
                    Some(VirtualKeyCode::D) => {
                        self.keys_pressed.right = pressed;
                        self.update_view_matrix();
                        self.updated = false;
                    }
                    _ => ()
                }
            }
            _ => ()
        }
    }

    pub fn moving(&self) -> bool {
        self.keys_pressed.left ||
            self.keys_pressed.right ||
            self.keys_pressed.up ||
            self.keys_pressed.down
    }

    pub fn update_view_matrix(&mut self) {
        let mut rot_matrix = Mat4::identity();

        rot_matrix = rotate_x(&rot_matrix, f32::to_radians(self.rotation.x));
        rot_matrix = rotate_y(&rot_matrix, f32::to_radians(self.rotation.y));
        rot_matrix = rotate_z(&rot_matrix, f32::to_radians(self.rotation.z));

        let trans_matrix = translate(&Mat4::identity(), &self.position);

        if self.camera_type == CameraType::FirstPerson {
            self.view_matrix = rot_matrix * trans_matrix;
        } else {
            self.view_matrix = trans_matrix * rot_matrix;
        }

        self.updated = true;
    }

    pub fn update(&mut self, delta_time: f32) {
        self.updated = false;

        if self.camera_type != CameraType::FirstPerson {
            return;
        }

        if !self.moving() {
            return;
        }

        let delta_time = delta_time / 1000.0;

        let cam_front = vec3(
            -self.rotation.x.to_radians().cos() * self.rotation.y.to_radians().sin(),
            self.rotation.x.to_radians().sin(),
            self.rotation.x.to_radians().cos() * self.rotation.y.to_radians().cos(),
        ).normalize();

        let move_speed = delta_time * self.movement_speed;

        if self.keys_pressed.up {
            self.position += cam_front * move_speed;
        }
        if self.keys_pressed.down {
            self.position -= cam_front * move_speed;
        }
        if self.keys_pressed.left {
            self.position -= cam_front.cross(&vec3(0.0, 1.0, 0.0)).normalize() * move_speed;
        }
        if self.keys_pressed.right {
            self.position += cam_front.cross(&vec3(0.0, 1.0, 0.0)).normalize() * move_speed;
        }

        self.update_view_matrix();
    }

    pub fn get_rotation(&self) -> Vec3 {
        self.rotation
    }

    pub fn translate(&mut self, delta: Vec3) {
        self.position += delta;
        self.update_view_matrix();
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        self.view_matrix
    }

    pub fn get_perspective_matrix(&self) -> Mat4 {
        let m = self.perspective_matrix;

        // correct for y pointing downwards
        // m[(1, 1)] *= -1.0;

        m
    }

    pub fn set_rotation(&mut self, rotation: Vec3) {
        self.rotation = rotation;
        self.update_view_matrix();
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
        self.update_view_matrix();
    }

    pub fn get_position(&mut self) -> Vec3 {
        self.position
    }

    pub fn set_perspective(&mut self, aspect: f32, fovy: f32, near: f32, far: f32) {
        self.perspective_matrix = nalgebra_glm::perspective(aspect, fovy, near, far);
    }

    pub fn rotate(&mut self, delta: Vec3) {
        self.rotation += delta;
    }
}
