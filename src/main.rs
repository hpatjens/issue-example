use nalgebra_glm::{cross, normalize, perspective_rh_zo};
use vulkano_win::VkSurfaceBuild;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState},
    device::{Device, DeviceExtensions},
    framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract},
    format::Format,
    image::{
        attachment::AttachmentImage,
        SwapchainImage,
    },
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        self, 
        Instance, 
        InstanceExtensions,
        PhysicalDevice
    },
    pipeline::{
        GraphicsPipeline, 
        viewport::Viewport,
    },
    swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive},
    swapchain,
    sync::{GpuFuture, FlushError},
    sync,
};
use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, DeviceEvent , WindowEvent, ElementState, MouseButton, MouseScrollDelta, VirtualKeyCode};
use std::{io, path::Path, sync::Arc, f32::consts::PI };

type Vec3 = nalgebra::Vector3<f32>;
type Mat4 = nalgebra::Matrix4<f32>;
type Point3 = nalgebra::Point3<f32>;

const DEPTH_FORMAT: Format = Format::D24Unorm_S8Uint;

#[derive(Default, Debug, Clone)]
pub struct Vertex { 
    pub position: [f32; 3],
    pub normal: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, normal);

pub struct Surface {
    pub index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub material: MeshMaterial,
}

pub enum MeshMaterial {
    Default,
    Reference(usize),
}
pub enum Material {
    OpaqueColor(Vec3),
}

pub struct Mesh {
    pub surface: Surface,
}

pub struct Model {
    pub materials: Vec<Material>,
    pub meshes: Vec<Mesh>,
}

#[derive(Debug)]
pub struct SphericalCamera {
    center: Vec3,
    radius: f32,
    phi: f32,
    theta: f32,
}

impl SphericalCamera {
    pub fn new() -> Self {
        SphericalCamera {
            center: Vec3::zeros(),
            radius: 2.0,
            phi: 0.0,
            theta: PI / 2.0,
        }
    }

    pub fn move_away(&mut self, dradius: f32) { 
        self.radius = 0.0f32.max(self.radius + dradius); 
    }

    pub fn rotate_up(&mut self, dtheta: f32) { 
        self.theta = PI.min(0.0f32.max(self.theta - dtheta));
    }

    pub fn rotate_right(&mut self, dphi: f32) { 
        self.phi = self.phi - dphi;
    }

    pub fn move_up(&mut self, delta: f32) {
        let eye = spherical_to_cartesian(self.radius, self.theta, self.phi);
        let right = cross(&Vec3::y_axis(), &eye);
        let camera_up = cross(&eye, &right);
        self.center += delta * normalize(&camera_up);
    }
    
    pub fn move_right(&mut self, delta: f32) {
        let eye = spherical_to_cartesian(self.radius, self.theta, self.phi);
        let right = cross(&Vec3::y_axis(), &eye);
        self.center -= delta * normalize(&right);
    }

    pub fn matrix(&self, viewport_size: (f32, f32)) -> Mat4 {
        let projection = perspective_rh_zo(
            viewport_size.0 / viewport_size.1,
            PI / 4.0,
            0.01,
            1000.0
        );
        
        let view = {
            let eye = spherical_to_cartesian(self.radius, self.theta, self.phi);
            Mat4::look_at_rh(&Point3::from(self.center + eye), &Point3::from(self.center), &-Vec3::y_axis())
        };
        
        projection * view
    }
}

fn spherical_to_cartesian(r: f32, theta: f32, phi: f32) -> Vec3 {
    Vec3::new(
        r * f32::sin(theta) * f32::cos(phi),
        r * f32::cos(theta),
        r * f32::sin(theta) * f32::sin(phi),
    )
}

pub fn import_model<P: AsRef<Path>>(device: &Arc<Device>, p: P) -> io::Result<Model> {
    let (models, materials) = match tobj::load_obj(p.as_ref()) {
        Ok(model) => model,
        Err(error) => panic!("{:?}", error),
    };
    Ok(Model {
        materials: {
            materials
                .iter()
                .map(|obj_material| Material::OpaqueColor(Vec3::new(obj_material.diffuse[0], obj_material.diffuse[1], obj_material.diffuse[2])))
                .collect()
        },
        meshes: {
            models
                .iter()
                .map(|obj_model| {
                    assert!(obj_model.mesh.positions.len() % 3 == 0, "position components have to be a multiple of 3.");
                    assert!(obj_model.mesh.normals.len() % 3 == 0, "normal components have to be a multiple of 3.");
                    let ref positions = obj_model.mesh.positions;
                    let ref normals = obj_model.mesh.normals;
                    let ref indices = obj_model.mesh.indices;
                    let vertices_count = positions.len() / 3;
                    let mut vertices = Vec::with_capacity(vertices_count);
                    for vertex_index in 0..vertices_count {
                        let index = vertex_index * 3;
                        vertices.push(Vertex {
                            position: [positions[index + 0], positions[index + 1], positions[index + 2]],
                            normal: [normals[index + 0], normals[index + 1], normals[index + 2]],
                        });
                    }
                    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertices.iter().cloned()).unwrap();
                    let index_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, indices.iter().cloned()).unwrap();
                    let material = match obj_model.mesh.material_id {
                        Some(material_id) => MeshMaterial::Reference(material_id),
                        None => MeshMaterial::Default,
                    };
                    let surface = Surface { vertex_buffer, index_buffer, material };
                    Mesh { surface }
                })
                .collect()
        }
    })
}

fn main() {
    let required_extensions = InstanceExtensions {
        ext_debug_utils: true,
        ..vulkano_win::required_extensions()
    };

    let layers = instance::layers_list().unwrap();
    for layer in layers {
        println!("{}", layer.name());
    }
    let layer = "VK_LAYER_LUNARG_standard_validation";
    let layers = vec![layer];

    let instance = Instance::new(None, &required_extensions, layers).unwrap();

    let severity = MessageSeverity {
        error: true,
        warning: true,
        information: true,
        verbose: true,
    };

    let ty = MessageType::all();

    let _debug_callback = DebugCallback::new(&instance, severity, ty, |msg| {
        let severity = if msg.severity.error {
            "error"
        } else if msg.severity.warning {
            "warning"
        } else if msg.severity.information {
            "information"
        } else if msg.severity.verbose {
            "verbose"
        } else {
            panic!("no-impl");
        };

        let ty = if msg.ty.general {
            "general"
        } else if msg.ty.validation {
            "validation"
        } else if msg.ty.performance {
            "performance" }
        else {
            panic!("no-impl");
        };

        println!("{} {} {}: {}", msg.layer_prefix, ty, severity, msg.description);
    }).ok();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();

    let queue_family = physical.queue_families().find(|&q| {
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };
    let (device, mut queues) = Device::new(
        physical, 
        physical.supported_features(), 
        &device_ext,
        [(queue_family, 0.5)].iter().cloned()
    ).unwrap();

    let queue = queues.next().unwrap();











    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            usage,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear
        ).unwrap()

    };

    let model = import_model(&device, Path::new("content/Cams650.obj")).unwrap();

    mod model_program {
        pub mod vs {
            vulkano_shaders::shader!{
                ty: "vertex",
                src: "
                #version 450

                layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 normal;

                layout(push_constant) uniform PushConstants {
                    vec4 color;
                    mat4 matrix;
                } push_constants;

                layout (location = 0) out vec3 out_normal;

                void main() {
                    gl_Position = push_constants.matrix * vec4(position, 1.0);
                    out_normal = normal;
                }
                "
            }
        }

        pub mod fs {
            vulkano_shaders::shader!{
                ty: "fragment",
                src: "
                #version 450

                layout(push_constant) uniform PushConstants {
                    vec4 color;
                    mat4 matrix;
                } push_constants;

                layout (location = 0) in vec3 in_normal;

                layout (location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(0.5* in_normal + vec3(0.5), 1.0);
                }
                "
            }
        }

        use std::sync::Arc;
        use vulkano::device::Device;

        pub struct Program {
            pub vs: vs::Shader,
            pub fs: fs::Shader,
        }

        impl Program {
            pub fn new(device: &Arc<Device>) -> Self {
                Program {
                    vs: vs::Shader::load(device.clone()).unwrap(),
                    fs: fs::Shader::load(device.clone()).unwrap(),
                }
            }
        }
    }

    let model_program = model_program::Program::new(&device);

    let surface_render_pass = Arc::new(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: DEPTH_FORMAT,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    ).unwrap());

    let surface_pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer()
        .vertex_shader(model_program.vs.main_entry_point(), ())
        .triangle_list()
        .front_face_clockwise()
        .cull_mode_back()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(model_program.fs.main_entry_point(), ())
        .depth_stencil_simple_depth()
        .depth_write(true)
        .render_pass(Subpass::from(surface_render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let mut dynamic_state = DynamicState { 
        line_width: None, 
        viewports: None, 
        scissors: None, 
        compare_mask: None, 
        write_mask: None, 
        reference: None
    };

    let mut framebuffers = window_size_dependent_setup(&device, &images, surface_render_pass.clone(), &mut dynamic_state);
    let mut spherical_camera = SphericalCamera::new();
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    let mut left_mouse_down = false;
    let mut middle_mouse_down = false;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreate_swapchain = true;
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                if input.state == ElementState::Pressed && input.virtual_keycode.map(|key| key == VirtualKeyCode::Escape).unwrap_or(false) {
                    *control_flow = ControlFlow::Exit;
                }
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state, button, .. }, .. } => {
                match button {
                    MouseButton::Left => left_mouse_down = state == ElementState::Pressed,
                    MouseButton::Middle => middle_mouse_down = state == ElementState::Pressed,
                    _ => (),
                }
            }
            Event::WindowEvent { event: WindowEvent::MouseWheel { delta, .. }, .. } => {
                match delta {
                    MouseScrollDelta::LineDelta(_x, y) => {
                        const ZOOM_SPEED: f32 = 10.0;
                        spherical_camera.move_away(-ZOOM_SPEED * y);
                    }
                    MouseScrollDelta::PixelDelta(_position) => {
                        unimplemented!()
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { .. }, .. } => {

            }
            Event::DeviceEvent { event: DeviceEvent::MouseMotion{ delta: (x, y) }, .. } => {
                if left_mouse_down {
                    const X_ROTATION_SPEED: f32 = 0.002;
                    const Y_ROTATION_SPEED: f32 = 0.001;
                    spherical_camera.rotate_right(X_ROTATION_SPEED * x as f32);
                    spherical_camera.rotate_up(Y_ROTATION_SPEED * y as f32);
                }
                if middle_mouse_down {
                    const MOVEMENT_SPEED: f32 = 0.1;
                    spherical_camera.move_right(-MOVEMENT_SPEED * x as f32);
                    spherical_camera.move_up(MOVEMENT_SPEED * y as f32);
                }
            }
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                let framebuffer_dimensions: [u32; 2] = surface.window().inner_size().into();

                if recreate_swapchain {
                    let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(framebuffer_dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e)
                    };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(&device, &new_images, surface_render_pass.clone(), &mut dynamic_state);
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    },
                    Err(e) => panic!("Failed to acquire next image: {:?}", e)
                };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec!([0.8, 0.85, 0.95, 1.0].into(), (1f32, 0u32).into());

                let matrix = {
                    let view_projection = {
                        let framebuffer_size = (framebuffer_dimensions[0] as f32, framebuffer_dimensions[1] as f32);
                        spherical_camera.matrix(framebuffer_size)
                    };
                    let world = Mat4::identity();
                    let na_matrix = view_projection * world;
    
                    [
                        [*na_matrix.index((0, 0)), *na_matrix.index((1, 0)), *na_matrix.index((2, 0)), *na_matrix.index((3, 0))],
                        [*na_matrix.index((0, 1)), *na_matrix.index((1, 1)), *na_matrix.index((2, 1)), *na_matrix.index((3, 1))],
                        [*na_matrix.index((0, 2)), *na_matrix.index((1, 2)), *na_matrix.index((2, 2)), *na_matrix.index((3, 2))],
                        [*na_matrix.index((0, 3)), *na_matrix.index((1, 3)), *na_matrix.index((2, 3)), *na_matrix.index((3, 3))],
                    ]
                };

                let surface_command_buffer = {
                    let auto_command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
                        .begin_render_pass(framebuffers[image_num].clone(), false, clear_values.clone()).unwrap();
                    let auto_command_buffer_builder = model.meshes
                        .iter()
                        .fold(auto_command_buffer_builder, |builder, mesh| {
                            let color = match mesh.surface.material {
                                MeshMaterial::Default => [0.5, 0.5, 0.5, 1.0],
                                MeshMaterial::Reference(material_id) => {
                                    let material = model.materials.get(material_id).expect("The material_id of the mesh must be correct.");
                                    match material {
                                        Material::OpaqueColor(color) => [color.x, color.y, color.z, 1.0],
                                    }
                                }
                            };
                            let push_constants = model_program::fs::ty::PushConstants { color, matrix };
                            builder.draw_indexed(surface_pipeline.clone(), &dynamic_state, mesh.surface.vertex_buffer.clone(), mesh.surface.index_buffer.clone(), (), push_constants).unwrap()
                        });
                    auto_command_buffer_builder
                        .end_render_pass().unwrap()
                        .build().unwrap()
                };

                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), surface_command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(Box::new(future) as Box<_>);
                    },
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                }
            },
            _ => ()
        }
    });
}

fn window_size_dependent_setup(
    device: &Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    let depth_buffer = AttachmentImage::transient(device.clone(), dimensions, DEPTH_FORMAT).unwrap();

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}