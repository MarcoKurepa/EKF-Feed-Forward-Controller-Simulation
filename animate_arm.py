import numpy as np
from TwoJointArm import TwoJointArm
import Trajectory
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

np.set_printoptions(linewidth=400)

arm = TwoJointArm()

fps = 20
dt = 1/fps
t0 = 0
tf = 10

def get_arm_joints(state):
    """Get the xy positions of all three robot joints (?) - base joint (at 0,0), elbow, end effector"""
    (joint_pos, eff_pos) = arm.get_lin_joint_pos(state[:2])
    x = np.array([0, joint_pos[0,0], eff_pos[0,0]])
    y = np.array([0, joint_pos[1,0], eff_pos[1,0]])
    return (x,y)

start_state = np.concatenate((arm.inv_kinematics(np.matrix([1.5, -1]).T, False), np.matrix([0,0]).T))
middle_state = np.concatenate((arm.inv_kinematics(np.matrix([1.5, 1]).T, True), np.matrix([0,0]).T))
end_state = np.concatenate((arm.inv_kinematics(np.matrix([-1.8, 1]).T, False), np.matrix([0,0]).T))

print(start_state)
print(middle_state)
print(end_state)

t0 = 0 # start time
t1 = 3 # time to arrive at middle_state
t2 = 4 # time to leave middle_state
t3 = 8 # time to arrive at end_state

traj1 = Trajectory.interpolate_states(t0, t1, start_state, middle_state)
traj2 = Trajectory.interpolate_states(t1, t2, middle_state, middle_state)
traj3 = Trajectory.interpolate_states(t2, t3, middle_state, end_state)

traj = traj1.append(traj2).append(traj3)

(xs, ys) = get_arm_joints(arm.state)
fig = plt.figure()
fig.set_size_inches(12,5)
ax = fig.add_subplot(4,5,(1,18))
ax.axis('square')
ax.grid(True)
ax.set_xlim(-arm.l1-arm.l2, arm.l1+arm.l2)
ax.set_ylim(-arm.l1-arm.l2, arm.l1+arm.l2)
target_line, arm_line, hat_line = ax.plot(xs, ys, 'b--o', xs, ys, 'r-o', xs, ys, 'g--o')
ax.legend([arm_line, target_line, hat_line], ["Current State", "Target State", "Estimated State"], loc='lower left')

fig2, ax2 = plt.subplots()
ax2.axis('square')
ax2.grid(True)
ax2.set_xlim(-np.pi, np.pi)
ax2.set_ylim(-np.pi, np.pi)
state_line, = ax2.plot([], [])


ax_1 = fig.add_subplot(4,5,4)
ax_2 = fig.add_subplot(4,5,5)
ax_3 = fig.add_subplot(4,5,14)
ax_4 = fig.add_subplot(4,5,15)
ax_1.set_xlim((t0, tf))
ax_2.set_xlim((t0, tf))
ax_3.set_xlim((t0, tf))
ax_4.set_xlim((t0, tf))
ax_1.grid(True)
ax_2.grid(True)
ax_3.grid(True)
ax_4.grid(True)
ax_1.yaxis.set_label("Voltage (V)")
ax_3.yaxis.set_label("Input Error (V)")
ax1_line1, ax1_line2 = ax_1.plot([], [], 'r', [], [], 'b')
ax2_line1, ax2_line2, ax2_line3 = ax_2.plot([], [], 'r', [], [], 'b', [], [], 'g')
ax3_line1, ax3_line2 = ax_3.plot([], [], 'r', [], [], 'b')
ax4_line1, ax4_line2, ax4_line3 = ax_4.plot([], [], 'r', [], [], 'b', [], [], 'g')

print("Starting sim...")
s = time.perf_counter()
sim_results = arm.simulate_with_ekf(traj, (t0, tf), initial_state = traj.sample(0)[:4])
e = time.perf_counter()
print("Finished sim")
print("Elapsed: %.02f us" % ((e-s)*10**6))
time_vec = sim_results.t
arm.last_controller_time = -10
arm.last_u = np.matrix([0,0]).T
voltage_log = sim_results.U
input_error_log = sim_results.U_err
true_pos = sim_results.X[:2,:]
enc_pos = sim_results.Xenc[:2,:]
est_pos = sim_results.Xhat[:2,:]
target_pos = sim_results.target[:2,:]
pos_err = sim_results.Xerr[:2,:]
Kcond = sim_results.Kcond
Acond = sim_results.Acond

ax_1.set_ylim((np.min(voltage_log), np.max(voltage_log)))
ax_2.set_ylim((np.min(est_pos-target_pos), np.max(est_pos-target_pos)))
ax_3.set_ylim((np.min(input_error_log), np.max(input_error_log)))
ax_4.set_ylim((np.min(est_pos-target_pos), np.max(est_pos-target_pos)))

ax_1.legend([ax1_line1, ax2_line2], ["J1 Voltage", "J2 Voltage"], loc='lower center', bbox_to_anchor = (0.5, -1))
ax_2.legend([ax2_line1, ax2_line2, ax2_line3], ["Encoder Err", "Est. Err", "True Err"], loc='lower center', bbox_to_anchor = (0.5, -1))
ax_3.legend([ax3_line1, ax3_line2], ["Input 1 Error", "Input 2 Error"], loc='lower center', bbox_to_anchor = (0.5, -1))
ax_4.legend([ax4_line1, ax4_line2, ax4_line3], ["Encoder Err", "Est. Err", "True Err"], loc='lower center', bbox_to_anchor = (0.5, -1))

def init():
    (xs, ys) = get_arm_joints(sim_results.X[:,0])
    arm_line.set_data(xs, ys)
    target_line.set_data(xs, ys)
    hat_line.set_data(xs, ys)
    ax.set_xlim(-arm.l1-arm.l2, arm.l1+arm.l2)
    ax.set_ylim(-arm.l1-arm.l2, arm.l1+arm.l2)
    ax1_line1.set_data([], [])
    ax1_line2.set_data([], [])
    ax2_line1.set_data([], [])
    ax2_line2.set_data([], [])
    ax2_line3.set_data([], [])
    ax3_line1.set_data([], [])
    ax3_line2.set_data([], [])
    ax4_line1.set_data([], [])
    ax4_line2.set_data([], [])
    ax4_line3.set_data([], [])
    state_line.set_data([], [])
    return arm_line, target_line, hat_line, ax1_line1, ax1_line2, ax2_line1, ax2_line2, ax2_line3, ax3_line1, ax3_line2, ax4_line1, ax4_line2, ax4_line3, state_line,

def animate(i):
    (xs, ys) = get_arm_joints(sim_results.X[:4,i])
    arm_line.set_data(xs, ys)
    (xs, ys) = get_arm_joints(traj.sample(sim_results.t[i])[:4,:])
    target_line.set_data(xs, ys)
    (xs, ys) = get_arm_joints(sim_results.Xhat[:4,np.min(i-1,0)])
    hat_line.set_data(xs, ys)
    ax.set_xlim(-arm.l1-arm.l2, arm.l1+arm.l2)
    ax.set_ylim(-arm.l1-arm.l2, arm.l1+arm.l2)

    ax1_line1.set_data(time_vec[:i], voltage_log[0,:i])
    ax1_line2.set_data(time_vec[:i], voltage_log[1,:i])

    ax2_line1.set_data(time_vec[:i], enc_pos[0,:i] - target_pos[0,:i])
    ax2_line2.set_data(time_vec[:i], est_pos[0,:i] - target_pos[0,:i])
    ax2_line3.set_data(time_vec[:i], true_pos[0,:i] - target_pos[0,:i])

    ax3_line1.set_data(time_vec[:i], input_error_log[0,:i])
    ax3_line2.set_data(time_vec[:i], input_error_log[1,:i])

    ax4_line1.set_data(time_vec[:i], enc_pos[1,:i] - target_pos[1,:i])
    ax4_line2.set_data(time_vec[:i], est_pos[1,:i] - target_pos[1,:i])
    ax4_line3.set_data(time_vec[:i], true_pos[1,:i] - target_pos[1,:i])

    theta1 = sim_results.X[0, :(i+1)]
    theta2 = sim_results.X[1, :(i+1)]
    state_line.set_data(theta1, theta2)

    return arm_line, target_line, hat_line, ax1_line1, ax1_line2, ax2_line1, ax2_line2, ax2_line3, ax3_line1, ax3_line2, ax4_line1, ax4_line2, ax4_line3, state_line,

nframes = len(sim_results.t)
anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = int(dt*1000), blit=False, repeat=True)


anim.save('sim_ekf_no_resizing_with_unmodelled.gif', writer='imagemagick')
