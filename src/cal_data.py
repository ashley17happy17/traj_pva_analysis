import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transbigdata as tbd

a = 6378137.0
e2 = 1 - np.power((6356752.3142 / 6378137.0), 2)

"""

# matplotlib show in wsl
1. Download VcXsrv(https://sourceforge.net/projects/vcxsrv/)
2. sudo apt-get update
3. python3 --version
4. sudo apt-get install python3.6-tk
5. pip install matplotlib
6. export DISPLAY=localhost:0.0

# Map Insertion
1. pip install -U transbigdata
2. Register and Set Plot Map Token (https://account.mapbox.com/)
    # tbd.set_mapboxtoken('pk.eyJ1IjoiYXNobGV5Y2hpdSIsImEiOiJjbTRheXloMjMwZGRtMmlzYXVpZDdhdnNiIn0.lYrtS23z8jm2ikg0T-JcbQ')
    # tbd.set_imgsavepath(r'/home/ashley/map/')

"""

def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    f_ref = config['Files']['GroundTruth']
    f_test = config['Files']['TestData']
    time = [float(config['CompareTime']['startTime']), float(config['CompareTime']['endTime'])]
    tunnel = float(config['CompareTime']['tunnel_length'])
    method = config['Interpolation']['method']
    out_path = config['Output']['Output_path']
    out_fig = config['Output']['Output_fig']
    col = []
    if (int(config['Comparison']['Position']) == 1):
        col.extend([1,2,3])
        print("extend pos")
    if (int(config['Comparison']['Velocity']) == 1):
        col.extend([4,5,6])
        print("extend vel")
    if (int(config['Comparison']['Attitude']) == 1):
        col.extend([7,8,9])
        print("extend att")
    return f_ref, f_test, time, tunnel, col, method, out_path, out_fig

def load_data(file):
    return pd.read_csv(file, sep=r'\s+', engine='python', header=None) #header='infer') #=

def linear_interp(inter_x, xf, yf):
    rowInt = len(inter_x)
    rowF = yf.shape[0]
    colF = yf.shape[1]
    inter_y = np.zeros((rowInt, colF))
    id_f_row = 0
    id_int_row = 0
    while (id_int_row < rowInt):
        if (inter_x[id_int_row] >= xf[id_f_row]) and (inter_x[id_int_row] <= xf[id_f_row+1]) and (id_f_row+1 < rowF):
            for idx_col in range(colF):
                check_yf = yf[id_f_row+1, idx_col] - yf[id_f_row, idx_col]
                if (idx_col >= 6) and (check_yf < -180):
                    check_yf += 360
                elif (idx_col >= 6) and (check_yf > 180):
                    check_yf -= 360
                inter_y[id_int_row, idx_col] = yf[id_f_row, idx_col] + (inter_x[id_int_row] - xf[id_f_row]) * (check_yf) / (xf[id_f_row+1] - xf[id_f_row])
            id_int_row += 1
        elif (id_f_row+1 >= rowF):
            print("Error: Please Check max index.")
            exit
        else:
            id_f_row = id_f_row + 1
    return inter_y

def interpolate_data(d_ref_array, d_test_array, time, col, method='linear'):
    x_ref = d_ref_array[:,0]
    y_ref = d_ref_array[:, col]
    x_test = d_test_array[:,0]
    y_test = d_test_array[:, col]
    time_known_s_flag = 0
    time_known_e_flag = 0

    # Find timestamp min and max
    if (time[0] == 0):
        min_t_test = np.min(x_test)
    else:
        if (time[0] < np.min(x_test)):
            min_t_test = np.min(x_test)
            print("[Warning]: Insert start time smaller than test data start time.")
        else:
            min_t_test = time[0]
            time_known_s_flag = 1
   
    if (time[1] == 0):
        max_t_test = np.max(x_test)
    else:
        if (time[1] > np.max(x_test)):
            max_t_test = np.max(x_test)
            print("[Warning]: Insert end time larger than test data end time.")
        else:
            max_t_test = time[1]
            time_known_e_flag = 1

    min_t_ref = np.min(x_ref) 
    max_t_ref = np.max(x_ref)

    # Find start and end interpolatation index
    if (min_t_test < min_t_ref):
        min_t_test_tmp = np.argmin(np.abs(x_test - min_t_ref))
        if (x_test[min_t_test_tmp] < x_ref[0]):
            min_t_test_tmp = min_t_test_tmp + 1
    else:
        if (time_known_s_flag != 1):
            min_t_test_tmp = np.argmin(x_ref)
        else:
            min_t_test_tmp  = np.argmin(np.abs(x_test - min_t_test))

    if (max_t_test > max_t_ref):
        max_t_test_tmp = np.argmin(np.abs(x_test - max_t_ref))
        if (x_test[max_t_test_tmp] > x_ref[x_ref.shape[0]-1]):
            max_t_test_tmp = max_t_test_tmp - 1
    else:
        if (time_known_e_flag != 1):
            max_t_test_tmp = np.argmax(x_ref)
        else:
            max_t_test_tmp = np.argmin(np.abs(x_test - max_t_test))
            
    idx_st = min_t_test_tmp
    idx_end = max_t_test_tmp + 1
    x_test_tmp = x_test[idx_st:idx_end]
    y_test_tmp = y_test[idx_st:idx_end]

    # Data Linear Interpolation
    y_ref_inter = np.zeros((max_t_test_tmp - min_t_test_tmp + 1, y_test.shape[1]))
    y_ref_inter = linear_interp(x_test_tmp, x_ref, y_ref)

    # Time and Data Combination
    inter_ref = np.hstack((x_test_tmp[:, np.newaxis], y_ref_inter))
    inter_test = np.hstack((x_test_tmp[:, np.newaxis], y_test_tmp))
    return inter_ref, inter_test

def geo2ecef(pos_LLA):
    c_lat = np.cos(pos_LLA[0])
    s_lat = np.sin(pos_LLA[0])
    c_lon = np.cos(pos_LLA[1])
    s_lon = np.sin(pos_LLA[1])
    Rn = a / np.sqrt(1 - e2 * np.power(pos_LLA[0], 2))
    if (1 - e2 * np.power(pos_LLA[0], 2)) < 0:
        print("warning\n")
    Rn_h = Rn + pos_LLA[2]
    x = Rn_h * c_lat * c_lon
    y = Rn_h * c_lat * s_lon
    z = (Rn * (1 - e2) + pos_LLA[2]) * s_lat
    pos_XYZ = [x, y, z]
    return pos_XYZ

def pos2dcm(lat, lon):
    s_lat = np.sin(lat)
    c_lat = np.cos(lat)
    s_lon = np.sin(lon)
    c_lon = np.cos(lon)
    C_ne = np.zeros((3, 3))
    C_ne[0,0] = -1.0 * s_lat * c_lon
    C_ne[0,1] = -1.0 * s_lon
    C_ne[0,2] = -1.0 * c_lat * c_lon
    C_ne[1,0] = -1.0 * s_lat * s_lon
    C_ne[1,1] = c_lon
    C_ne[1,2] = -1.0 * c_lat * s_lon
    C_ne[2,0] = c_lat
    C_ne[2,1] = 0.0
    C_ne[2,2] = -1.0 * s_lat
    return C_ne

def geo2ned(data_LLA, first_point):
    pos_XYZ_f = geo2ecef(first_point)
    pos_NED = np.copy(data_LLA)
    rows, cols = data_LLA.shape
    for idx in range(rows):
        pos_XYZ_t = geo2ecef(data_LLA[idx, 1:4])
        diff_XYZ = np.array(pos_XYZ_t) - np.array(pos_XYZ_f)
        C_ne = pos2dcm(data_LLA[idx, 1], data_LLA[idx, 2])
        C_en = C_ne.T
        pos_NED[idx, 1:4] = (C_en @ diff_XYZ).flatten()
    return pos_NED

def att2dcm(roll, pitch, head):
    yaw = -head
    s_rol = np.sin(roll)
    c_rol = np.cos(roll)
    s_pit = np.sin(pitch)
    c_pit = np.cos(pitch)
    s_yaw = np.sin(yaw)
    c_yaw = np.cos(yaw)
    C_nb = np.zeros((3, 3))
    C_nb[0,0] = c_pit * c_yaw
    C_nb[0,1] = s_rol * s_pit * c_yaw - c_rol * s_yaw
    C_nb[0,2] = c_yaw * s_pit * c_yaw + s_rol * s_yaw
    C_nb[1,0] = c_pit * s_yaw
    C_nb[1,1] = s_rol * s_pit * s_yaw + c_rol * c_yaw
    C_nb[1,2] = c_rol * s_pit * s_yaw - s_rol * c_yaw
    C_nb[2,0] = -1.0 *  s_pit
    C_nb[2,1] = s_rol * c_pit
    C_nb[2,2] = c_rol * c_pit
    return C_nb

def pos_n2b(pos_nf, att):
    row, col = pos_nf.shape
    pos_bf = np.zeros((row, col))
    for idx_i in range(row):
        C_nb = att2dcm(att[idx_i, 0], att[idx_i, 1], att[idx_i, 2])
        data = np.array(pos_nf[idx_i, :])
        pos_bf[idx_i, :] = (C_nb @ data).flatten()           
    return pos_bf

def calculate_rmse(error, col):
    id = np.copy(col)
    error_squa = np.square(error)
    error_mean_squa = np.mean(error_squa, axis=id)
    rmse = np.sqrt(error_mean_squa)
    return rmse

def calculate_ERR_MAE(err, err_H, err_3D, err_bf, isATT):
    mae_NED = np.mean(np.abs(err), axis=0)
    mae_H = np.mean(np.abs(err_H))
    mae_3D = np.mean(np.abs(err_3D))
    if (not isATT):
        mae_bf = np.mean(np.abs(err_bf), axis=0)
        err_MAE = [mae_NED[1], mae_NED[0], mae_NED[2], mae_H, mae_3D, mae_bf[0], mae_bf[1]]
    else:
        err_MAE = [mae_NED[0], mae_NED[1], mae_NED[2]]
    return err_MAE

def calculate_ERR_MAX(err, err_H, err_3D, err_bf, isATT):
    max_NED = np.zeros(3)
    max_bf = np.zeros(3)
    err_MAX_id = np.zeros(3)
    for col in range(0, 3):
        max_idx = np.argmax(np.abs(err[:, col]))
        max_NED[col] = err[max_idx, col]
        max_bf_idx = np.argmax(np.abs(err_bf[:, col]))
        max_bf[col] = err_bf[max_bf_idx, col]
        err_MAX_id[col] = max_idx
    
    max_H = np.max(np.abs(err_H))
    max_3D = np.max(np.abs(err_3D))
    if (not isATT):  
        err_MAX = [max_NED[1], max_NED[0], max_NED[2], max_H, max_3D, max_bf[0], max_bf[1]]
    else:
        err_MAX = [max_NED[0], max_NED[1], max_NED[2]]
    return err_MAX, err_MAX_id

def calculate_ERR_STD(err, err_H, err_3D, err_bf, isATT):
    std_NED = np.std(err, axis=0)
    std_H = np.std(err_H)
    std_3D = np.std(err_3D)
    if (not isATT):
        std_bf = np.std(err_bf, axis=0)
        err_STD = [std_NED[1], std_NED[0], std_NED[2], std_H, std_3D, std_bf[0], std_bf[1]]
    else:
        err_STD = [std_NED[0], std_NED[1], std_NED[2]]
    return err_STD

def calculate_ERR_RMSE(err, err_H, err_3D, err_bf, isATT):
    rmse_NED = calculate_rmse(err, 0)
    rmse_H = calculate_rmse(err_H, 0)
    rmse_3D = calculate_rmse(err_3D, 0)
    if (not isATT):
        rmse_bf = calculate_rmse(err_bf, 0)
        err_RMSE = [rmse_NED[1], rmse_NED[0], rmse_NED[2], rmse_H, rmse_3D, rmse_bf[0], rmse_bf[1]]
    else:
        err_RMSE = [rmse_NED[0], rmse_NED[1], rmse_NED[2]]
    return err_RMSE

def plot_figure_err(data_x, data_y, max_id, title):
    plt.title(title)
    plt.ylabel('Error')
    max_x = data_x[int(max_id)]
    max_y = data_y[int(max_id)]
    plt.plot(data_x, data_y, color='blue', linewidth=2)
    annotation_text = f'x={max_x:.2f}, y={max_y:.2f}'
    plt.annotate(annotation_text, xy=(max_x, max_y), xytext=(max_x+15, max_y), fontsize=15)

def error_calculation(pos_xyz_ref, pos_xyz_test, out_path, out_fig, data_col):
    pos_err= np.zeros((4,7))
    vel_err = np.zeros((4,7))
    att_err = np.zeros((4,3))
    att_ref = np.deg2rad(pos_xyz_ref[:,7:10])

    # === POS ERROR ===
    pos_err_NED = pos_xyz_test[:,1:4] - pos_xyz_ref[:,1:4]

    rows, cols = pos_err_NED.shape
    if (len(data_col) > 3):
        pos_err_bf = pos_n2b(pos_err_NED, att_ref)
    elif (len(data_col) == 3):
        pos_err_bf = np.zeros((rows, 3))
    pos_err_2D = np.sqrt(np.square(pos_err_NED[:,0]) + np.square(pos_err_NED[:,1]))
    pos_err_3D = np.sqrt(np.square(pos_err_NED[:,0]) + np.square(pos_err_NED[:,1]) + np.square(pos_err_NED[:,2]))
    pos_err_2D_re = pos_err_2D.reshape(-1,1)
    pos_err_3D_re = pos_err_3D.reshape(-1,1)
    pos_err_time = np.concatenate([pos_xyz_test[:,0:1], pos_err_NED, pos_err_2D_re, pos_err_3D_re, pos_err_bf[:,0:2]], axis=1)

    np.savetxt("../data/output.csv", pos_err_time, delimiter=",", fmt="%.3f")

    # Error Calculation
    pos_err_MAE = calculate_ERR_MAE(pos_err_NED, pos_err_2D, pos_err_3D, pos_err_bf, 0)
    pos_err_MAX, pos_Max_id = calculate_ERR_MAX(pos_err_NED, pos_err_2D, pos_err_3D, pos_err_bf, 0)
    pos_err_STD = calculate_ERR_STD(pos_err_NED, pos_err_2D, pos_err_3D, pos_err_bf, 0)
    pos_err_RMSE = calculate_ERR_RMSE(pos_err_NED, pos_err_2D, pos_err_3D, pos_err_bf, 0)
    # Form POS_ENU_ERR
    pos_err = [pos_err_MAE, pos_err_MAX, pos_err_STD, pos_err_RMSE]

    if (len(data_col) > 3):
        # === VEL ERROR ===
        vel_err_NED = pos_xyz_test[:,4:7] - pos_xyz_ref[:,4:7]
        vel_err_bf = pos_n2b(vel_err_NED, att_ref)
        vel_err_2D = np.sqrt(np.square(vel_err_NED[:,0]) + np.square(vel_err_NED[:,1]))
        vel_err_3D = np.sqrt(np.square(vel_err_NED[:,0]) + np.square(vel_err_NED[:,1]) + np.square(vel_err_NED[:,2]))
        # Error Calculation
        vel_err_MAE = calculate_ERR_MAE(vel_err_NED, vel_err_2D, vel_err_3D, vel_err_bf, 0)
        vel_err_MAX, vel_Max_id = calculate_ERR_MAX(vel_err_NED, vel_err_2D, vel_err_3D, vel_err_bf, 0)
        vel_err_STD = calculate_ERR_STD(vel_err_NED, vel_err_2D, vel_err_3D, vel_err_bf, 0)
        vel_err_RMSE = calculate_ERR_RMSE(vel_err_NED, vel_err_2D, vel_err_3D, vel_err_bf, 0)
        # Form VEL_ERR_ENU
        vel_err = [vel_err_MAE, vel_err_MAX, vel_err_STD, vel_err_RMSE]

        # === ATT ERROR ===
        att_err_NED = pos_xyz_test[:,7:10] - pos_xyz_ref[:,7:10]
        row, col = att_err_NED.shape
        for i in range(row):
            for j in range(col):
                if (att_err_NED[i,j] < -180):
                    att_err_NED[i,j] = (att_err_NED[i,j] + 360)
                elif (att_err_NED[i,j] > 180):
                    att_err_NED[i,j] = (att_err_NED[i,j] - 360)
        
        att_err_bf = np.zeros((row, col))
        att_err_2D = np.sqrt(np.square(att_err_NED[:,0]) + np.square(att_err_NED[:,1]))
        att_err_3D = np.sqrt(np.square(att_err_NED[:,0]) + np.square(att_err_NED[:,1]) + np.square(att_err_NED[:,2]))
        # Error Calculation
        att_err_MAE = calculate_ERR_MAE(att_err_NED, att_err_2D, att_err_3D, att_err_bf, 1)
        att_err_MAX, att_Max_id = calculate_ERR_MAX(att_err_NED, att_err_2D, att_err_3D, att_err_bf, 1)
        att_err_STD = calculate_ERR_STD(att_err_NED, att_err_2D, att_err_3D, att_err_bf, 1)
        att_err_RMSE = calculate_ERR_RMSE(att_err_NED, att_err_2D, att_err_3D, att_err_bf, 1)
        # Form ATT_ERR_ENU
        att_err = [att_err_MAE, att_err_MAX, att_err_STD, att_err_RMSE]

    # Figure Plot
    fig_title = ["E(m)", "N(m)", "U(m)", "E(m/s)", "N(m/s)", "U(m/s)", "Roll(deg)", "Pitch(deg)", "Head(deg)"]
    data_x = pos_xyz_test[:, 0]

    for i in range (3):
        data_y = pos_err_NED[:, i]
        max_id = pos_Max_id[i]
        title = "Error Distribution: Position " + fig_title[i]
        fig = plt.figure(num=2, figsize=(15,10))
        plt.subplot(3,1,i+1)
        plot_figure_err(data_x, data_y, max_id, title)
        if (out_fig):
            plt.savefig(out_path+'fig2-1_errPosition.png')

    if (len(data_col) > 3):
        for i in range (3):
            data_y = vel_err_NED[:, i]
            max_id = vel_Max_id[i]
            title = "Error Distribution: Velocity " + fig_title[3+i]
            fig = plt.figure(num=3, figsize=(15,10))
            plt.subplot(3,1,i+1)
            plot_figure_err(data_x, data_y, max_id, title)
            if (out_fig):
                plt.savefig(out_path+'fig2-2_errVelocity.png')

        for i in range (3):
            data_y = att_err_NED[:, i]
            max_id = att_Max_id[i]
            title = "Error Distribution: Attitude " + fig_title[6+i]
            fig = plt.figure(num=4, figsize=(15,10))
            plt.subplot(3,1,i+1)
            plot_figure_err(data_x, data_y, max_id, title)
            if (out_fig):
                plt.savefig(out_path+'fig2-3_errAttitude.png')

    return pos_err, vel_err, att_err

# 繪製誤差分布圖
def plot_error_distribution(error):
    for col, values in error.items():
        plt.hist(values, bins=50, alpha=0.7, label=f'Column {col}')
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.legend()
    plt.show()

def main():
    config_path = "config_for_cal.ini"
    f_ref, f_test, time, tunnel, data_col, method, out_path, out_fig = read_config(config_path)

    if len(data_col) == 0:
        print("No accuracy comparison item is selected.", end='\n')
        exit
    
    d_ref = load_data(f_ref)
    d_test = load_data(f_test)
    print(">> Read Reference Data: ", f_ref)
    print(">> Read Test Data: ", f_test)
    print(">> Start time: ", time[0])
    print(">> End time: ", time[1])
    print(">> Finish Loading Data", end='\n')

    # Transform Data to array format
    d_ref_array = d_ref.to_numpy()
    d_test_array = d_test.to_numpy()

    #Transform LLA deg2rad
    d_ref_rad = np.copy(d_ref_array)
    d_test_rad = np.copy(d_test_array)
    d_ref_rad[:,1:3] = np.deg2rad(d_ref_array[:,1:3])
    d_test_rad[:,1:3] = np.deg2rad(d_test_array[:,1:3])
    d_test_rad[:,3:4] = d_test_rad[:,3:4] #  + 17.1  Geoid Separation Compensation(From Ublox GGA)

    # Data Interpolation
    inter_ref, inter_test = interpolate_data(d_ref_rad, d_test_rad, time, data_col, method)
    inter_ref_deg = np.rad2deg(inter_ref[:,1:3])
    inter_test_deg = np.rad2deg(inter_test[:,1:3])
    # inter_ref = d_ref_rad
    # inter_test = d_test_rad

    # Transform Data from Lat,Lon,H to LLF
    first_point = inter_ref[0,1:4]
    pos_xyz_ref = geo2ned(inter_ref, first_point)
    first_point = inter_ref[0,1:4]
    pos_xyz_test = geo2ned(inter_test, first_point)

    # Calculate Reference Data Distance Accumulation
    ref_dist_2D = 0
    ref_dist_3D = 0
    for idx in range(len(pos_xyz_ref)-1):
        x1, y1, z1 = pos_xyz_ref[idx, 1:4]
        x2, y2, z2 = pos_xyz_ref[idx+1, 1:4]
        ref_dist_2D += np.sqrt((x1-x2)**2 + (y1-y2)**2)
        ref_dist_3D += np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    # Precision Calculation
    pos_err, vel_err, att_err = error_calculation(pos_xyz_ref, pos_xyz_test, out_path, out_fig, data_col)
    
    print("======================= Result Evaluation =======================")
    print("== Position (m) ==")
    print("        E       N       U       2D       3D   AlongT   CrossT")
    print("MAE    " + "   ".join(f"{val:.3f}" for val in pos_err[0]))
    print("MAX    " + "   ".join(f"{val:.3f}" for val in pos_err[1]))
    print("STD    " + "   ".join(f"{val:.3f}" for val in pos_err[2]))
    print("RMSE   " + "   ".join(f"{val:.3f}" for val in pos_err[3]))
    print("== Velocity (m/s) ==")
    print("        E       N       U       2D       3D   AlongT   CrossT")
    print("MAE    " + "   ".join(f"{val:.3f}" for val in vel_err[0]))
    print("MAX    " + "   ".join(f"{val:.3f}" for val in vel_err[1]))
    print("STD    " + "   ".join(f"{val:.3f}" for val in vel_err[2]))
    print("RMSE   " + "   ".join(f"{val:.3f}" for val in vel_err[3]))
    print("== Attitude (deg) ==")
    print("       Roll    Pitch    Head")
    print("MAE    " + "   ".join(f"{val:.3f}" for val in att_err[0]))
    print("MAX    " + "   ".join(f"{val:.3f}" for val in att_err[1]))
    print("STD    " + "   ".join(f"{val:.3f}" for val in att_err[2]))
    print("RMSE   " + "   ".join(f"{val:.3f}" for val in att_err[3]))

    if (tunnel != 0):
        print("== Error Drift Over ==")
        # DT = pos_err[1][3] / tunnel *100
        time_diff = time[1] - time[0]
        DT_2D = pos_err[1][3] / ref_dist_2D *100
        TT_2D = pos_err[1][3] / time_diff
        DT_3D = pos_err[1][4] / ref_dist_3D *100
        TT_3D = pos_err[1][4] / time_diff
        # print(f"Distance Travelled (DT, %)    {DT:.3f} ({tunnel:.1f}m)")
        print(f"2D Distance Travelled (DT, %)    {DT_2D:.3f} ({ref_dist_2D:.1f}m)")
        print(f"2D Time Travelled (TT, m/s)    {TT_2D:.3f} ({time_diff:.1f}sec)")
        print(f"3D Distance Travelled (DT, %)    {DT_3D:.3f} ({ref_dist_3D:.1f}m)")
        print(f"3D Time Travelled (TT, m/s)    {TT_3D:.3f} ({time_diff:.1f}sec)")

    print("=================================================================")

    # Set Plot Map Token
    # tbd.set_mapboxtoken('pk.eyJ1IjoiYXNobGV5Y2hpdSIsImEiOiJjbTRheXloMjMwZGRtMmlzYXVpZDdhdnNiIn0.lYrtS23z8jm2ikg0T-JcbQ')
    # tbd.set_imgsavepath(r'/home/ashley/map/')

    # Plot Trajectory
    if (out_fig):
        fig1 = plt.figure(num=1,figsize=(15,10))
        buf = 0.001
        min_lat = inter_ref_deg[np.argmin(inter_ref_deg[:,0]), 0]
        min_lon = inter_ref_deg[np.argmin(inter_ref_deg[:,1]), 1]
        max_lat = inter_ref_deg[np.argmax(inter_ref_deg[:,0]), 0]
        max_lon = inter_ref_deg[np.argmax(inter_ref_deg[:,1]), 1]
        bounds = [min_lon-buf, min_lat-buf, max_lon+buf, max_lat+buf]
        ax =plt.subplot(111)
        plt.sca(ax)
        tbd.plot_map(plt,bounds,zoom = 16,style = 3)
        # [style]
        # 1: streets, 2: outdoors, 3: satellite, 4: light, 5:dark, 6: light-ch, 9: terrain, 11: light(no comment), 12: dark(no comment)

        tbd.plotscale(ax,bounds = bounds,textsize = 10,compasssize = 2,accuracy = 5,rect = [0.06,0.03])
        plt.axis('off')
        plt.xlim(bounds[0],bounds[2])
        plt.ylim(bounds[1],bounds[3])

        plt.title('Trajectory Comparison')
        plt.xlabel('Longtitude')
        plt.ylabel('Latitude')
        plt.plot(inter_ref_deg[:,1], inter_ref_deg[:,0], color='red', linewidth=2)
        plt.plot(inter_test_deg[:,1], inter_test_deg[:,0], color='cyan', linewidth=2)
        plt.legend(["ref","test"])
        if (out_fig):
            plt.savefig(out_path+'fig1_traj.png')
        plt.show()

if __name__ == "__main__":
    main()
