#Author: Brandon J. Woodard
#Institution: Brown University

import numpy as np
from scipy import interpolate, integrate, signal, optimize, stats
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import csv
import pandas as pd
import numba as nb
from sklearn.metrics import mean_squared_error
def read_transmit(filename):
    data = np.genfromtxt(filename,delimiter=',', skip_header=1)
    data = np.matrix(data)
    transmit_y = data[:, 0]
    transmit_y = transmit_y[np.logical_not(np.isnan(transmit_y))]
    transmit_y = transmit_y.tolist()[0]
    
    receive_y = data[:, 1]
    receive_y = receive_y[np.logical_not(np.isnan(receive_y))]
    receive_y = receive_y.tolist()[0]
    
    return transmit_y, receive_y

def read_signal_file(filename, file_type):
    """
    

    Parameters
    ----------
    filename : string
        filename of wave csv to open

    Returns
    -------
    transmit_y : list
        list of transmit amplitudes
    receive_y : list
        list of receive amplitudes

    """
    
    if file_type == 1:
        data = np.genfromtxt(filename,delimiter=':', skip_header=1)
        data = np.matrix(data)
        transmit_y = data[:, 0]
        transmit_y = transmit_y[np.logical_not(np.isnan(transmit_y))]
        transmit_y = transmit_y.tolist()[0]
        
        receive_y = data[:, 1]
        receive_y = receive_y[np.logical_not(np.isnan(receive_y))]
        receive_y = receive_y.tolist()[0]
        
        return transmit_y, receive_y
    if file_type == 2:
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)      
        data = np.matrix(data) 
        
        xs = data[:, 2]
        xs = xs[np.logical_not(np.isnan(xs))]
        xs = xs.tolist()[0]
        
        ys = data[:, 1]
        ys = ys[np.logical_not(np.isnan(ys))]
        ys = ys.tolist()[0]
        
        return xs[::-1], ys[::-1]
    if file_type == 3:
        xs = []
        ys = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=':')
            header = next(reader)
            for row in reader:
                current_xs = []
                current_ys = []
                delta = (float(row[3]) - float(row[4]))/float(row[8])
                rx = row[10].split(',')
                rx = list(map(float, rx))[:-1]
                for i in range(len(rx)):
                    current_xs.append(float(row[3])+i*delta)
                    current_ys.append(rx[i])
                xs.append(current_xs)
                ys.append(current_ys)
        
        return xs, ys
    
    if file_type == 4:
        
        orig_file = open(filename, 'r')
        new_filename = 'newfile.csv'
        new_file = open(new_filename,'w')
        lines = []
        found_quote = 0
        for line in orig_file:
            current_line = ""
            for i in range(len(line)):
                if line[i] == '"':
                    current_line += '"'
                    found_quote = (found_quote +1 ) % 2
                elif line[i] == ',':
                    if found_quote == 0:
                        current_line += ":"
                    else:
                        current_line += line[i]
                else:
                    current_line += line[i]
            lines.append(current_line)
        new_file.writelines(lines)
        
        return_xs = []
        return_ys = []
        transmit_xs = []
        transmit_ys = []
        with open(new_filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=':')
            header = next(reader)
            for row in reader:
                current_transmit_xs = []
                current_transmit_ys = []
                current_return_xs = []
                current_return_ys = []
                delta = (float(row[3]) - float(row[4]))/float(row[9])
                rx = row[13].split(',')
                tx = row[14].split(',')
                rx = list(map(float, rx))[:-1]
                tx = list(map(float, tx))[:-1]
                for i in range(len(rx)):                                        
                    current_return_xs.append(float(row[3])/3.3356 + i)
                    current_return_ys.append(rx[i])
                
                for i in range(len(tx)):
                    current_transmit_xs.append(i)
                    current_transmit_ys.append(tx[i])
                
                
                return_xs.append(current_return_xs)
                return_ys.append(current_return_ys)
                transmit_xs.append(current_transmit_xs)
                transmit_ys.append(current_transmit_ys)
        
        return return_xs, return_ys, transmit_xs, transmit_ys

    if file_type == 5:
        
        orig_file = open(filename, 'r')
        new_filename = 'newfile.csv'
        new_file = open(new_filename,'w')
        lines = []
        found_quote = 0
        for line in orig_file:
            current_line = ""
            for i in range(len(line)):
                if line[i] == '"':
                    current_line += '"'
                    found_quote = (found_quote +1 ) % 2
                elif line[i] == ',':
                    if found_quote == 0:
                        current_line += ":"
                    else:
                        current_line += line[i]
                else:
                    current_line += line[i]
            lines.append(current_line)
        new_file.writelines(lines)
        
        return_xs = []
        return_ys = []
        transmit_xs = []
        transmit_ys = []
        
        geo = []
        
        ms = []
        
        with open(new_filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=':')
            header = next(reader)
            for row in reader:
                current_transmit_xs = []
                current_transmit_ys = []
                current_return_xs = []
                current_return_ys = []
                delta = (float(row[3]) - float(row[4]))/float(row[11])
                bin_0 = row[3]
                last_bin = row[4]
                alt_instr = row[9]
                rx = row[15].split(',')
                tx = row[16].split(',')
                rx = list(map(float, rx))[:-1]
                tx = list(map(float, tx))[:-1]
                for i in range(len(rx)):                                             
                    current_return_xs.append((float(row[9]) - float(row[3]))*2/3.3356 + i)
                    current_return_ys.append(rx[i])
                
                for i in range(len(tx)):
                    current_transmit_xs.append(i)
                    current_transmit_ys.append(tx[i])
                
                elevation_angle = float(row[8])
                ms.append(np.tan(np.pi/2 - elevation_angle))
                
                
                return_xs.append(current_return_xs)
                return_ys.append(current_return_ys)
                transmit_xs.append(current_transmit_xs)
                transmit_ys.append(current_transmit_ys)
                
                g = row[14].split(" ")
                geo.append((g[1][1:], g[2][:-2]))
        
        return return_xs, return_ys, transmit_xs, transmit_ys, ms, geo, bin_0, last_bin, alt_instr
    
def extend_to_match(s1, s2):
    """
    

    Parameters
    ----------
    s1 : list of floats
        the signal to be extended
    s2 : list of floats
        the signal to match

    Returns
    -------
    the extended version of s1 to match the length of s2

    """
    
    diff = len(s2) - len(s1)
    for i in range(diff):
        if i % 2 == 0:
            s1 = [s1[0]] + s1
        else:
            s1 = s1 + [s1[-1]]

    return s1

def align_peak(s1, s2):
    """
    

    Parameters
    ----------
    s1 : list of floats
        the signal to align to s2
    s2 : TYPE
        the signal to be aligned to

    Returns
    -------
    returns s1 with peak aligned to s2

    """
    peak_s1 = max(s1)
    peak_s2 = max(s2)
    diff = s2.index(peak_s2) - s1.index(peak_s1)
    if diff > 0:
        for i in range(diff):
            s1 = [s1[-1]] + s1[:-1]
    else:
        for i in range(-diff):
            s1 = s1[1:] + [s1[0]]
            
    return s1

def find_vertical_offset(s1, tol=20):
    """
    

    Parameters
    ----------
    s1 : list of floats
        signal to find baseline of.
    tol : float, optional
        an offset to look for baseline points in. The default is 20.

    Returns
    -------
    float
        offset of baseline from 0

    """
    m = min(s1)
    s1 = np.array(s1)
    return sum(s1[s1 < m+tol])/len(s1[s1 < m+tol])

def find_horizontal_offset(x, y):
    """
    

    Parameters
    ----------
    x : list of floats
        DESCRIPTION.
    y : list of floats
        DESCRIPTION.

    Returns
    -------
    float
        horizontal offset of signal y, relative to peak being centered at zero.

    """
    h_index = np.array(y).argmax()
    return x[h_index]

def shift_to_zero(x, y):
    """
    

    Parameters
    ----------
    x : list of floats
        DESCRIPTION.
    y : list of floats
        amplitudes for corresponding x.

    Returns
    -------
    list of floats
        x's shifted such that the peak of the signal is at 0.

    """
    m_index = np.array(y).argmax()
    return list(np.array(x) - x[m_index])


def eval_spline(x, spline, xmin, xmax):
    if x >= xmin and x <= xmax:
        return interpolate.splev(x, spline)
    elif x < xmin:
        return interpolate.splev(xmin, spline)
    else:
        return interpolate.splev(xmax, spline)
    
def eval_spline_list(xs, spline, xmin, xmax):
    results = []
    for x in xs:
        if x >= xmin and x <= xmax:
            results.append(interpolate.splev(x, spline))
        elif x < xmin:
            return interpolate.splev(xmin, spline)
        else:
            return interpolate.splev(xmax, spline)
    return np.array(results)


def convolve_with_gaussian(xs, signal, m = 0.87, N=200):
    """
    

    Parameters
    ----------
    xs : list of floats
        times for corresponding amplitudes in signal
    s1 : list of floats
        wave to convolve with
    m : float, optional
        the slope. The default is 1.31.

    Returns
    -------
    integ_result : list of floats
        integral of s1 with gaussian

    """
    def integrand(x, m , t, xmin, xmax):
        return (1/(np.sqrt(2*np.pi)*np.abs(m)))*eval_spline(x, spline, xmin, xmax)*np.e**(-((t-x)**2)/(2*m**2))
    
    def R(t, spline, m, xmin, xmax):
        delta = (xmax-xmin)/N
        integ_result = 0
        for i in range(N):
            integ_result += integrate.quad(integrand, (xmin)+i*delta, (xmin)+(i+1)*delta, args=(m, t, xmin, xmax), epsabs=1.49e-02, epsrel=1.49e-02)[0]
        return integ_result
    spline = interpolate.splrep(xs, signal, k=3)
    
    return list(map(lambda t: R(t, spline, m, xs[0], xs[-1]), xs))

def find_start_of_last_mode(x1, s1, height=300, distance=50):
    peaks = find_peaks(s1, height = height, distance = distance)
    print("Peaks: ", peaks)
    m = 1e20
    m_index = 0
    
    if len(peaks[0]) < 2:
        return 0
    
    for i in range(peaks[0][-2], peaks[0][-1]):
        if s1[i] < m:
            m = s1[i]
            m_index = i
            
    return m_index

def fit_to_other_signal(x1, s1, x2, s2, params=np.concatenate((2*np.random.rand(2)-1, 2*np.random.rand(2))), numruns=1):
    x1 = np.array(x1)
    x2 = np.array(x2)
    s1 = np.array(s1)
    s2 = np.array(s2)
    spl = interpolate.splrep(x1, s1, k=3)
    
    v_offset = find_vertical_offset(s1)
    
    def residuals(params):
        vs = params[0]
        hs = params[1]
        vsc = params[2]
        hsc = params[3]
        #hsc = 1
        considered_xs = hsc*x1+hs
        considered_ys = vsc*eval_spline_list(x1, spl, x1[0], x1[-1]) + vs
        newspl = interpolate.splrep(considered_xs, considered_ys, k=3)
        considered_xs = x1
        considered_ys = eval_spline_list(x1, newspl, x1[0], x1[-1])
        return np.abs(s2-considered_ys)
    
    runs = np.zeros((1,2))
    for i in range(numruns):
        print(i+1, "/", numruns)
        opt = optimize.least_squares(residuals, params, bounds = ([-np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]), ftol=None, xtol=1e-15, max_nfev=1000, verbose=0)
        params = opt.x
        runs = np.vstack((runs, (opt.x, opt.optimality)))
    
    runs = runs[1:]
    #print(runs)
    vals = min(runs[:,1])
    index = runs[:,1].tolist().index(vals)
    params = runs[index,0]
    #print(params, vals)
    vs = params[0]
    hs = params[1]
    vsc = params[2]
    hsc = params[3]
    print("Best vs: ", vs)
    print("Best hs: ", hs)
    print("Best vsc: ", vsc)
    print("Best hsc: ", hsc)
    
    considered_xs = hsc*x1+hs
    considered_ys = vsc*eval_spline_list(x1, spl, x1[0], x1[-1])+vs
    newspl = interpolate.splrep(considered_xs, considered_ys, k=3)
    considered_xs = x1
    considered_ys = interpolate.splev(considered_xs, newspl)
    return considered_xs, considered_ys, params

def compute_FWHM(xs, ys, height=300, distance=50):
    
    peaks = find_peaks(ys, height = height, distance = distance)    
    baseline = find_vertical_offset(ys)   
    half_height_y = (ys[peaks[0][0]] + baseline)/2
    
    spline = interpolate.splrep(xs, ys, k=3)
    
    def S(x):
        return interpolate.splev(x, spline) - half_height_y
    
    delta = 0.01
    tol = 0.1
    
    x1 = None
    x2 = None
    
    N = int((xs[-1] - xs[0])/delta)
    
    for i, x in enumerate(np.linspace(xs[0], xs[-1], N)):
        if i % 1000 == 0:
            print(str(i) + "/"+ str(N))
        if np.abs(S(x)) < tol:
            if x1 != None:
                x2 = x
            else:
                x1 = x
        if x1 != None and x2 != None and np.abs(interpolate.splev(x, spline) - baseline) < tol:
            break

    return (x2-x1), x1, x2, half_height_y

### Test signals ###
def bump_with_error(xs, s):
    def gaussian(x, s):
        return (1/(s*np.sqrt(2*np.pi)))*np.e**(-(x**2)/(2*s**2)) + 0.001*np.random.random()
    return list(map(lambda x : gaussian(x,s), xs))

def packet(xs, t):
    r = []
    c = 2
    for x in xs:
        r.append(np.exp(-(x-c*t)**2)*(np.cos(2*np.pi*(x-c*t)/(2)) + complex(0, 1)*np.sin(2*np.pi*(x-c*t)/(2))))
    return r
###

def return_troughs(x1, s1, height=300, distance=50,prominence=100):
    peaks = find_peaks(s1, height = height, distance = distance,prominence = prominence)
    #if len(peaks[0]) < 2:
    #    return 0

if __name__ == '__main__':
    

    def test8():
        filename = 'la_selva_snip_1.csv'
        bonneville_return_xs, bonneville_return_ys, bonneville_transmit_xs, bonneville_transmit_ys, ms, geo, bin_0, last_bin, alt_instr = read_signal_file(filename, 5)
        result = []
        for i in range(1, 2):
            receive_xs = bonneville_return_xs[i]            
            receive_ys = bonneville_return_ys[i]
            transmit_ys = bonneville_transmit_ys[i]
            transmit_xs = bonneville_transmit_xs[i]
            orig_receive_xs = receive_xs

            fig, ax = plt.subplots()
            print("Length of transmit xs")
            print(len(transmit_xs))
            
            transmit_ys = extend_to_match(transmit_ys, receive_ys)
            transmit_xs = receive_xs

            #transmit_xs = transmit_xs - transmit_xs[0]

            transmit_vertical_offset = find_vertical_offset(transmit_ys)
            transmit_horizontal_offset = find_horizontal_offset(transmit_xs, transmit_ys)
            receive_vertical_offset = find_vertical_offset(receive_ys)
            receive_horizontal_offset = find_horizontal_offset(receive_xs, receive_ys)
            
            print("***")
            print(receive_horizontal_offset)
            print(receive_vertical_offset)
            print(ms[i])
            print("***")
         
            #Note: I don't actually want to align peaks
            transmit_ys = align_peak(transmit_ys, receive_ys)
            receive_xs = shift_to_zero(receive_xs, receive_ys)
            
            gaussian_xs = receive_xs
            gaussian_ys = list(map(lambda x:  1/(np.abs(6.62)*np.sqrt(np.pi*2))*np.exp(-(1/2)*(x/6.62)**2), gaussian_xs))
            
            
            #Note: labeling this as recieve xs is throwing me off a bit
            #ax.plot(transmit_xs, transmit_ys, label='transmit', c='black')
            #ax.plot(receive_xs, receive_ys, label='receive')
            convolved_ys = convolve_with_gaussian(transmit_xs, transmit_ys-find_vertical_offset(transmit_ys), ms[i])
            #convolved_ys = convolve_with_gaussian(transmit_xs, transmit_ys, ms[i])
            #ax.plot(receive_xs, convolved_ys, label='convolved', c = 'green')
            index_of_trough = find_start_of_last_mode(receive_xs, receive_ys)
            
            print("Trough: ", index_of_trough)
            
            
            # Peak of receive
            receive_peak_x_index = np.array(receive_ys[index_of_trough:]).argmax()
            receive_peak_x = receive_xs[index_of_trough:][receive_peak_x_index]
            receive_peak_y = receive_ys[index_of_trough:][receive_peak_x_index]
            #ax.scatter(receive_peak_x+receive_horizontal_offset, receive_peak_y, marker='x')
            
            fit_xs, fit_ys, params = fit_to_other_signal(orig_receive_xs[index_of_trough:], convolved_ys[index_of_trough:], orig_receive_xs[index_of_trough:], receive_ys[index_of_trough:], params=[receive_ys[index_of_trough:][0], 0, 1, 1], numruns=1)
            print("Fit_xs", fit_xs[0], fit_xs[-1])
            fit_gaussian_xs, fit_gaussian_ys, gaussian_params = fit_to_other_signal(orig_receive_xs[index_of_trough:], gaussian_ys[index_of_trough:], receive_xs[index_of_trough:], receive_ys[index_of_trough:], params = [receive_ys[index_of_trough:][0], 0, max(receive_ys[index_of_trough:])*np.random.random(), 1], numruns=5)
            
            right_base,left_base = return_troughs(fit_xs[index_of_trough],fit_ys)
            #print(right_base,left_base)
            
            ax.plot(orig_receive_xs[index_of_trough:right_base[0]], fit_ys, c='red', label = 'Model R(t)')
            ax.plot(orig_receive_xs[index_of_trough:], fit_gaussian_ys, c='green', label = 'Gaussian')
            mid_point_x = ((fit_xs[left_base[0]] + fit_xs[right_base[0]])/2)
            spl = interpolate.splrep(np.array(fit_xs[left_base[0]:right_base[0]]), np.array(fit_ys[left_base[0]:right_base[0]]),k=3)
            mid_point_y = interpolate.splev(mid_point_x,spl)
            #Peak of model
            model_peak_x_index = np.array(fit_ys).argmax()
            model_peak_x = orig_receive_xs[index_of_trough + model_peak_x_index]
            model_peak_y = fit_ys[model_peak_x_index]
            ax.scatter(model_peak_x, model_peak_y, marker='x')

            print("Length of receive xs")
            print(len(receive_xs))
            ax.set_xlabel("Nanoseconds")
            ax.set_ylabel("ADC (counts)")

            ax.plot(orig_receive_xs, np.array(receive_ys), c='blue', label = "Receive (GEDI)")
            print(orig_receive_xs[0])
            #ax.invert_xaxis()
            #ax.plot(np.array(fit_xs)+receive_horizontal_offset, np.array(fit_ys), c='red', label = "Model R(t)")
            #ax.plot(np.array(fit_gaussian_xs)+receive_horizontal_offset, np.array(fit_gaussian_ys), c = 'green', label='Gaussian Fit')
            """
            #Finding FWHM
            fit_fwhm, fwhm_x1, fwhm_x2, fwhm_y = compute_FWHM(orig_receive_xs[index_of_trough:], fit_ys)
            receive_fwhm, receive_fwhm_x1, receive_fwhm_x2, receive_fwhm_y = compute_FWHM(orig_receive_xs, receive_ys)
            ax.plot([fwhm_x1, fwhm_x2],[fwhm_y, fwhm_y], c='purple', label='Model R(t) FWHM: {}'.format(round(fit_fwhm,3)))
            ax.plot([receive_fwhm_x1, receive_fwhm_x2],[receive_fwhm_y, receive_fwhm_y], c='purple', label = 'Receive FWHM {}'.format(round(receive_fwhm,3)))
            print("FWHM:")
            print("Fit: ", fit_fwhm, fwhm_x1, fwhm_x2, fwhm_y)
            print("Receive: ", receive_fwhm, receive_fwhm_x1, receive_fwhm_x2, receive_fwhm_y)
            """
            #rms = mean_squared_error(y_actual, y_predicted, squared=False)
            
            
            
            # print("Fit FWHM: ", fit_fwhm)
            # print("Receive FWHM: ", receive_fwhm)
            
            
            
            #ax.set_xlim(orig_receive_xs[-1], orig_receive_xs[0])
            #ax.set_xlim((np.array(receive_xs)+receive_horizontal_offset)[0], (np.array(receive_xs)+receive_horizontal_offset)[-1])
            
            ax.legend()
            plt.show()
            
            print("R(t): ", chisquare(fit_ys, receive_ys[index_of_trough:])[0])
            print("Gaussian(t): ", chisquare(fit_gaussian_ys, receive_ys[index_of_trough:])[0])
            #Note: params[1] + offset should be accurate because if it's neg it will just be subtracted
            result.append([params[0], params[1], params[2], params[3], model_peak_x, model_peak_y, receive_peak_x, receive_peak_y, geo[i], chisquare(fit_ys, receive_ys[index_of_trough:])[0]])
        columns = ['vs', 'hs', 'vsc', 'hsc', 'model peak x', 'model peak y', 'receive peak x', 'receive peak y', 'geo', 'chisquare']
        df = pd.DataFrame(result, columns = columns)
        print(df)
    
    test8()