from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import datetime

from sklearn.ensemble import GradientBoostingClassifier

from .forms import DoctorForm
from .models import *
from django.contrib.auth import authenticate, login, logout
import numpy as np
import pandas as pd
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from django.http import HttpResponse
# Create your views here.

def Home(request):
    return render(request,'carousel.html')

def Admin_Home(request):
    dis = Search_Data.objects.all()
    pat = Patient.objects.all()
    doc = Doctor.objects.all()
    feed = Feedback.objects.all()

    d = {'dis':dis.count(),'pat':pat.count(),'doc':doc.count(),'feed':feed.count()}
    return render(request,'admin_home.html',d)

@login_required(login_url="login")
def assign_status(request,pid):
    doctor = Doctor.objects.get(id=pid)
    if doctor.status == 1:
        doctor.status = 2
        messages.success(request, 'Selected doctor are successfully withdraw his approval.')
    else:
        doctor.status = 1
        messages.success(request, 'Selected doctor are successfully approved.')
    doctor.save()
    return redirect('view_doctor')

@login_required(login_url="login")
def User_Home(request):
    return render(request,'patient_home.html')

@login_required(login_url="login")
def Doctor_Home(request):
    return render(request,'doctor_home.html')

def About(request):
    return render(request,'about.html')

def Contact(request):
    return render(request,'contact.html')


def Gallery(request):
    return render(request,'gallery.html')


def Login_User(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        sign = ""
        if user:
            try:
                sign = Patient.objects.get(user=user)
            except:
                pass
            if sign:
                login(request, user)
                error = "pat1"
            else:
                pure=False
                try:
                    pure = Doctor.objects.get(status=1,user=user)
                except:
                    pass
                if pure:
                    login(request, user)
                    error = "pat2"
                else:
                    login(request, user)
                    error="notmember"
        else:
            error="not"
    d = {'error': error}
    return render(request, 'login.html', d)

def Login_admin(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        if user.is_staff:
            login(request, user)
            error="pat"
        else:
            error="not"
    d = {'error': error}
    return render(request, 'admin_login.html', d)

def Signup_User(request):
    error = ""
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        u = request.POST['uname']
        e = request.POST['email']
        p = request.POST['pwd']
        d = request.POST['dob']
        con = request.POST['contact']
        add = request.POST['add']
        type = request.POST['type']
        im = request.FILES['image']
        dat = datetime.date.today()
        user = User.objects.create_user(email=e, username=u, password=p, first_name=f,last_name=l)
        if type == "Patient":
            Patient.objects.create(user=user,contact=con,address=add,image=im,dob=d)
        else:
            Doctor.objects.create(dob=d,image=im,user=user,contact=con,address=add,status=2)
        error = "create"
    d = {'error':error}
    return render(request,'register.html',d)

def Logout(request):
    logout(request)
    return redirect('home')

@login_required(login_url="login")
def Change_Password(request):
    sign = 0
    user = User.objects.get(username=request.user.username)
    error = ""
    if not request.user.is_staff:
        try:
            sign = Patient.objects.get(user=user)
            if sign:
                error = "pat"
        except:
            sign = Doctor.objects.get(user=user)
    terror = ""
    if request.method=="POST":
        n = request.POST['pwd1']
        c = request.POST['pwd2']
        o = request.POST['pwd3']
        if c == n:
            u = User.objects.get(username__exact=request.user.username)
            u.set_password(n)
            u.save()
            terror = "yes"
        else:
            terror = "not"
    d = {'error':error,'terror':terror,'data':sign}
    return render(request,'change_password.html',d)


def preprocess_inputs(df, scaler):
    df = df.copy()
    # Split df into X and y
    y = df['target'].copy()
    X = df.drop('target', axis=1).copy()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y


def prdict_heart_disease(list_data):
    #print(list_data)
    csv_file = Admin_Helath_CSV.objects.get(id=1)
    data =[[63,1,3,145,233,1,0,150,0,2.3,0,0,1,1],
[37,1,2,130,250,0,1,187,0,3.5,0,0,2,1],
[41,0,1,130,204,0,0,172,0,1.4,2,0,2,1],
[56,1,1,120,236,0,1,178,0,0.8,2,0,2,1],
[57,0,0,120,354,0,1,163,1,0.6,2,0,2,1],
[57,1,0,140,192,0,1,148,0,0.4,1,0,1,1],
[56,0,1,140,294,0,0,153,0,1.3,1,0,2,1],
[44,1,1,120,263,0,1,173,0,0,2,0,3,1],
[52,1,2,172,199,1,1,162,0,0.5,2,0,3,1],
[57,1,2,150,168,0,1,174,0,1.6,2,0,2,1],
[54,1,0,140,239,0,1,160,0,1.2,2,0,2,1],
[48,0,2,130,275,0,1,139,0,0.2,2,0,2,1],
[49,1,1,130,266,0,1,171,0,0.6,2,0,2,1],
[64,1,3,110,211,0,0,144,1,1.8,1,0,2,1],
[58,0,3,150,283,1,0,162,0,1,2,0,2,1],
[50,0,2,120,219,0,1,158,0,1.6,1,0,2,1],
[58,0,2,120,340,0,1,172,0,0,2,0,2,1],
[66,0,3,150,226,0,1,114,0,2.6,0,0,2,1],
[43,1,0,150,247,0,1,171,0,1.5,2,0,2,1],
[69,0,3,140,239,0,1,151,0,1.8,2,2,2,1],
[59,1,0,135,234,0,1,161,0,0.5,1,0,3,1],
[44,1,2,130,233,0,1,179,1,0.4,2,0,2,1],
[42,1,0,140,226,0,1,178,0,0,2,0,2,1],
[61,1,2,150,243,1,1,137,1,1,1,0,2,1],
[40,1,3,140,199,0,1,178,1,1.4,2,0,3,1],
[71,0,1,160,302,0,1,162,0,0.4,2,2,2,1],
[59,1,2,150,212,1,1,157,0,1.6,2,0,2,1],
[51,1,2,110,175,0,1,123,0,0.6,2,0,2,1],
[65,0,2,140,417,1,0,157,0,0.8,2,1,2,1],
[53,1,2,130,197,1,0,152,0,1.2,0,0,2,1],
[41,0,1,105,198,0,1,168,0,0,2,1,2,1],
[65,1,0,120,177,0,1,140,0,0.4,2,0,3,1],
[44,1,1,130,219,0,0,188,0,0,2,0,2,1],
[54,1,2,125,273,0,0,152,0,0.5,0,1,2,1],
[51,1,3,125,213,0,0,125,1,1.4,2,1,2,1],
[46,0,2,142,177,0,0,160,1,1.4,0,0,2,1],
[54,0,2,135,304,1,1,170,0,0,2,0,2,1],
[54,1,2,150,232,0,0,165,0,1.6,2,0,3,1],
[65,0,2,155,269,0,1,148,0,0.8,2,0,2,1],
[65,0,2,160,360,0,0,151,0,0.8,2,0,2,1],
[51,0,2,140,308,0,0,142,0,1.5,2,1,2,1],
[48,1,1,130,245,0,0,180,0,0.2,1,0,2,1],
[45,1,0,104,208,0,0,148,1,3,1,0,2,1],
[53,0,0,130,264,0,0,143,0,0.4,1,0,2,1],
[39,1,2,140,321,0,0,182,0,0,2,0,2,1],
[52,1,1,120,325,0,1,172,0,0.2,2,0,2,1],
[44,1,2,140,235,0,0,180,0,0,2,0,2,1],
[47,1,2,138,257,0,0,156,0,0,2,0,2,1],
[53,0,2,128,216,0,0,115,0,0,2,0,0,1],
[53,0,0,138,234,0,0,160,0,0,2,0,2,1],
[51,0,2,130,256,0,0,149,0,0.5,2,0,2,1],
[66,1,0,120,302,0,0,151,0,0.4,1,0,2,1],
[62,1,2,130,231,0,1,146,0,1.8,1,3,3,1],
[44,0,2,108,141,0,1,175,0,0.6,1,0,2,1],
[63,0,2,135,252,0,0,172,0,0,2,0,2,1],
[52,1,1,134,201,0,1,158,0,0.8,2,1,2,1],
[48,1,0,122,222,0,0,186,0,0,2,0,2,1],
[45,1,0,115,260,0,0,185,0,0,2,0,2,1],
[34,1,3,118,182,0,0,174,0,0,2,0,2,1],
[57,0,0,128,303,0,0,159,0,0,2,1,2,1],
[71,0,2,110,265,1,0,130,0,0,2,1,2,1],
[54,1,1,108,309,0,1,156,0,0,2,0,3,1],
[52,1,3,118,186,0,0,190,0,0,1,0,1,1],
[41,1,1,135,203,0,1,132,0,0,1,0,1,1],
[58,1,2,140,211,1,0,165,0,0,2,0,2,1],
[35,0,0,138,183,0,1,182,0,1.4,2,0,2,1],
[51,1,2,100,222,0,1,143,1,1.2,1,0,2,1],
[45,0,1,130,234,0,0,175,0,0.6,1,0,2,1],
[44,1,1,120,220,0,1,170,0,0,2,0,2,1],
[62,0,0,124,209,0,1,163,0,0,2,0,2,1],
[54,1,2,120,258,0,0,147,0,0.4,1,0,3,1],
[51,1,2,94,227,0,1,154,1,0,2,1,3,1],
[29,1,1,130,204,0,0,202,0,0,2,0,2,1],
[51,1,0,140,261,0,0,186,1,0,2,0,2,1],
[43,0,2,122,213,0,1,165,0,0.2,1,0,2,1],
[55,0,1,135,250,0,0,161,0,1.4,1,0,2,1],
[51,1,2,125,245,1,0,166,0,2.4,1,0,2,1],
[59,1,1,140,221,0,1,164,1,0,2,0,2,1],
[52,1,1,128,205,1,1,184,0,0,2,0,2,1],
[58,1,2,105,240,0,0,154,1,0.6,1,0,3,1],
[41,1,2,112,250,0,1,179,0,0,2,0,2,1],
[45,1,1,128,308,0,0,170,0,0,2,0,2,1],
[60,0,2,102,318,0,1,160,0,0,2,1,2,1],
[52,1,3,152,298,1,1,178,0,1.2,1,0,3,1],
[42,0,0,102,265,0,0,122,0,0.6,1,0,2,1],
[67,0,2,115,564,0,0,160,0,1.6,1,0,3,1],
[68,1,2,118,277,0,1,151,0,1,2,1,3,1],
[46,1,1,101,197,1,1,156,0,0,2,0,3,1],
[54,0,2,110,214,0,1,158,0,1.6,1,0,2,1],
[58,0,0,100,248,0,0,122,0,1,1,0,2,1],
[48,1,2,124,255,1,1,175,0,0,2,2,2,1],
[57,1,0,132,207,0,1,168,1,0,2,0,3,1],
[52,1,2,138,223,0,1,169,0,0,2,4,2,1],
[54,0,1,132,288,1,0,159,1,0,2,1,2,1],
[45,0,1,112,160,0,1,138,0,0,1,0,2,1],
[53,1,0,142,226,0,0,111,1,0,2,0,3,1],
[62,0,0,140,394,0,0,157,0,1.2,1,0,2,1],
[52,1,0,108,233,1,1,147,0,0.1,2,3,3,1],
[43,1,2,130,315,0,1,162,0,1.9,2,1,2,1],
[53,1,2,130,246,1,0,173,0,0,2,3,2,1],
[42,1,3,148,244,0,0,178,0,0.8,2,2,2,1],
[59,1,3,178,270,0,0,145,0,4.2,0,0,3,1],
[63,0,1,140,195,0,1,179,0,0,2,2,2,1],
[42,1,2,120,240,1,1,194,0,0.8,0,0,3,1],
[50,1,2,129,196,0,1,163,0,0,2,0,2,1],
[68,0,2,120,211,0,0,115,0,1.5,1,0,2,1],
[69,1,3,160,234,1,0,131,0,0.1,1,1,2,1],
[45,0,0,138,236,0,0,152,1,0.2,1,0,2,1],
[50,0,1,120,244,0,1,162,0,1.1,2,0,2,1],
[50,0,0,110,254,0,0,159,0,0,2,0,2,1],
[64,0,0,180,325,0,1,154,1,0,2,0,2,1],
[57,1,2,150,126,1,1,173,0,0.2,2,1,3,1],
[64,0,2,140,313,0,1,133,0,0.2,2,0,3,1],
[43,1,0,110,211,0,1,161,0,0,2,0,3,1],
[55,1,1,130,262,0,1,155,0,0,2,0,2,1],
[37,0,2,120,215,0,1,170,0,0,2,0,2,1],
[41,1,2,130,214,0,0,168,0,2,1,0,2,1],
[56,1,3,120,193,0,0,162,0,1.9,1,0,3,1],
[46,0,1,105,204,0,1,172,0,0,2,0,2,1],
[46,0,0,138,243,0,0,152,1,0,1,0,2,1],
[64,0,0,130,303,0,1,122,0,2,1,2,2,1],
[59,1,0,138,271,0,0,182,0,0,2,0,2,1],
[41,0,2,112,268,0,0,172,1,0,2,0,2,1],
[54,0,2,108,267,0,0,167,0,0,2,0,2,1],
[39,0,2,94,199,0,1,179,0,0,2,0,2,1],
[34,0,1,118,210,0,1,192,0,0.7,2,0,2,1],
[47,1,0,112,204,0,1,143,0,0.1,2,0,2,1],
[67,0,2,152,277,0,1,172,0,0,2,1,2,1],
[52,0,2,136,196,0,0,169,0,0.1,1,0,2,1],
[74,0,1,120,269,0,0,121,1,0.2,2,1,2,1],
[54,0,2,160,201,0,1,163,0,0,2,1,2,1],
[49,0,1,134,271,0,1,162,0,0,1,0,2,1],
[42,1,1,120,295,0,1,162,0,0,2,0,2,1],
[41,1,1,110,235,0,1,153,0,0,2,0,2,1],
[41,0,1,126,306,0,1,163,0,0,2,0,2,1],
[49,0,0,130,269,0,1,163,0,0,2,0,2,1],
[60,0,2,120,178,1,1,96,0,0,2,0,2,1],
[62,1,1,128,208,1,0,140,0,0,2,0,2,1],
[57,1,0,110,201,0,1,126,1,1.5,1,0,1,1],
[64,1,0,128,263,0,1,105,1,0.2,1,1,3,1],
[51,0,2,120,295,0,0,157,0,0.6,2,0,2,1],
[43,1,0,115,303,0,1,181,0,1.2,1,0,2,1],
[42,0,2,120,209,0,1,173,0,0,1,0,2,1],
[67,0,0,106,223,0,1,142,0,0.3,2,2,2,1],
[76,0,2,140,197,0,2,116,0,1.1,1,0,2,1],
[70,1,1,156,245,0,0,143,0,0,2,0,2,1],
[44,0,2,118,242,0,1,149,0,0.3,1,1,2,1],
[60,0,3,150,240,0,1,171,0,0.9,2,0,2,1],
[44,1,2,120,226,0,1,169,0,0,2,0,2,1],
[42,1,2,130,180,0,1,150,0,0,2,0,2,1],
[66,1,0,160,228,0,0,138,0,2.3,2,0,1,1],
[71,0,0,112,149,0,1,125,0,1.6,1,0,2,1],
[64,1,3,170,227,0,0,155,0,0.6,1,0,3,1],
[66,0,2,146,278,0,0,152,0,0,1,1,2,1],
[39,0,2,138,220,0,1,152,0,0,1,0,2,1],
[58,0,0,130,197,0,1,131,0,0.6,1,0,2,1],
[47,1,2,130,253,0,1,179,0,0,2,0,2,1],
[35,1,1,122,192,0,1,174,0,0,2,0,2,1],
[58,1,1,125,220,0,1,144,0,0.4,1,4,3,1],
[56,1,1,130,221,0,0,163,0,0,2,0,3,1],
[56,1,1,120,240,0,1,169,0,0,0,0,2,1],
[55,0,1,132,342,0,1,166,0,1.2,2,0,2,1],
[41,1,1,120,157,0,1,182,0,0,2,0,2,1],
[38,1,2,138,175,0,1,173,0,0,2,4,2,1],
[38,1,2,138,175,0,1,173,0,0,2,4,2,1],
[67,1,0,160,286,0,0,108,1,1.5,1,3,2,0],
[67,1,0,120,229,0,0,129,1,2.6,1,2,3,0],
[62,0,0,140,268,0,0,160,0,3.6,0,2,2,0],
[63,1,0,130,254,0,0,147,0,1.4,1,1,3,0],
[53,1,0,140,203,1,0,155,1,3.1,0,0,3,0],
[56,1,2,130,256,1,0,142,1,0.6,1,1,1,0],
[48,1,1,110,229,0,1,168,0,1,0,0,3,0],
[58,1,1,120,284,0,0,160,0,1.8,1,0,2,0],
[58,1,2,132,224,0,0,173,0,3.2,2,2,3,0],
[60,1,0,130,206,0,0,132,1,2.4,1,2,3,0],
[40,1,0,110,167,0,0,114,1,2,1,0,3,0],
[60,1,0,117,230,1,1,160,1,1.4,2,2,3,0],
[64,1,2,140,335,0,1,158,0,0,2,0,2,0],
[43,1,0,120,177,0,0,120,1,2.5,1,0,3,0],
[57,1,0,150,276,0,0,112,1,0.6,1,1,1,0],
[55,1,0,132,353,0,1,132,1,1.2,1,1,3,0],
[65,0,0,150,225,0,0,114,0,1,1,3,3,0],
[61,0,0,130,330,0,0,169,0,0,2,0,2,0],
[58,1,2,112,230,0,0,165,0,2.5,1,1,3,0],
[50,1,0,150,243,0,0,128,0,2.6,1,0,3,0],
[44,1,0,112,290,0,0,153,0,0,2,1,2,0],
[60,1,0,130,253,0,1,144,1,1.4,2,1,3,0],
[54,1,0,124,266,0,0,109,1,2.2,1,1,3,0],
[50,1,2,140,233,0,1,163,0,0.6,1,1,3,0],
[41,1,0,110,172,0,0,158,0,0,2,0,3,0],
[51,0,0,130,305,0,1,142,1,1.2,1,0,3,0],
[58,1,0,128,216,0,0,131,1,2.2,1,3,3,0],
[54,1,0,120,188,0,1,113,0,1.4,1,1,3,0],
[60,1,0,145,282,0,0,142,1,2.8,1,2,3,0],
[60,1,2,140,185,0,0,155,0,3,1,0,2,0],
[59,1,0,170,326,0,0,140,1,3.4,0,0,3,0],
[46,1,2,150,231,0,1,147,0,3.6,1,0,2,0],
[67,1,0,125,254,1,1,163,0,0.2,1,2,3,0],
[62,1,0,120,267,0,1,99,1,1.8,1,2,3,0],
[65,1,0,110,248,0,0,158,0,0.6,2,2,1,0],
[44,1,0,110,197,0,0,177,0,0,2,1,2,0],
[60,1,0,125,258,0,0,141,1,2.8,1,1,3,0],
[58,1,0,150,270,0,0,111,1,0.8,2,0,3,0],
[68,1,2,180,274,1,0,150,1,1.6,1,0,3,0],
[62,0,0,160,164,0,0,145,0,6.2,0,3,3,0],
[52,1,0,128,255,0,1,161,1,0,2,1,3,0],
[59,1,0,110,239,0,0,142,1,1.2,1,1,3,0],
[60,0,0,150,258,0,0,157,0,2.6,1,2,3,0],
[49,1,2,120,188,0,1,139,0,2,1,3,3,0],
[59,1,0,140,177,0,1,162,1,0,2,1,3,0],
[57,1,2,128,229,0,0,150,0,0.4,1,1,3,0],
[61,1,0,120,260,0,1,140,1,3.6,1,1,3,0],
[39,1,0,118,219,0,1,140,0,1.2,1,0,3,0],
[61,0,0,145,307,0,0,146,1,1,1,0,3,0],
[56,1,0,125,249,1,0,144,1,1.2,1,1,2,0],
[43,0,0,132,341,1,0,136,1,3,1,0,3,0],
[62,0,2,130,263,0,1,97,0,1.2,1,1,3,0],
[63,1,0,130,330,1,0,132,1,1.8,2,3,3,0],
[65,1,0,135,254,0,0,127,0,2.8,1,1,3,0],
[48,1,0,130,256,1,0,150,1,0,2,2,3,0],
[63,0,0,150,407,0,0,154,0,4,1,3,3,0],
[55,1,0,140,217,0,1,111,1,5.6,0,0,3,0],
[65,1,3,138,282,1,0,174,0,1.4,1,1,2,0],
[56,0,0,200,288,1,0,133,1,4,0,2,3,0],
[54,1,0,110,239,0,1,126,1,2.8,1,1,3,0],
[70,1,0,145,174,0,1,125,1,2.6,0,0,3,0],
[62,1,1,120,281,0,0,103,0,1.4,1,1,3,0],
[35,1,0,120,198,0,1,130,1,1.6,1,0,3,0],
[59,1,3,170,288,0,0,159,0,0.2,1,0,3,0],
[64,1,2,125,309,0,1,131,1,1.8,1,0,3,0],
[47,1,2,108,243,0,1,152,0,0,2,0,2,0],
[57,1,0,165,289,1,0,124,0,1,1,3,3,0],
[55,1,0,160,289,0,0,145,1,0.8,1,1,3,0],
[64,1,0,120,246,0,0,96,1,2.2,0,1,2,0],
[70,1,0,130,322,0,0,109,0,2.4,1,3,2,0],
[51,1,0,140,299,0,1,173,1,1.6,2,0,3,0],
[58,1,0,125,300,0,0,171,0,0,2,2,3,0],
[60,1,0,140,293,0,0,170,0,1.2,1,2,3,0],
[77,1,0,125,304,0,0,162,1,0,2,3,2,0],
[35,1,0,126,282,0,0,156,1,0,2,0,3,0],
[70,1,2,160,269,0,1,112,1,2.9,1,1,3,0],
[59,0,0,174,249,0,1,143,1,0,1,0,2,0],
[64,1,0,145,212,0,0,132,0,2,1,2,1,0],
[57,1,0,152,274,0,1,88,1,1.2,1,1,3,0],
[56,1,0,132,184,0,0,105,1,2.1,1,1,1,0],
[48,1,0,124,274,0,0,166,0,0.5,1,0,3,0],
[56,0,0,134,409,0,0,150,1,1.9,1,2,3,0],
[66,1,1,160,246,0,1,120,1,0,1,3,1,0],
[54,1,1,192,283,0,0,195,0,0,2,1,3,0],
[69,1,2,140,254,0,0,146,0,2,1,3,3,0],
[51,1,0,140,298,0,1,122,1,4.2,1,3,3,0],
[43,1,0,132,247,1,0,143,1,0.1,1,4,3,0],
[62,0,0,138,294,1,1,106,0,1.9,1,3,2,0],
[67,1,0,100,299,0,0,125,1,0.9,1,2,2,0],
[59,1,3,160,273,0,0,125,0,0,2,0,2,0],
[45,1,0,142,309,0,0,147,1,0,1,3,3,0],
[58,1,0,128,259,0,0,130,1,3,1,2,3,0],
[50,1,0,144,200,0,0,126,1,0.9,1,0,3,0],
[62,0,0,150,244,0,1,154,1,1.4,1,0,2,0],
[38,1,3,120,231,0,1,182,1,3.8,1,0,3,0],
[66,0,0,178,228,1,1,165,1,1,1,2,3,0],
[52,1,0,112,230,0,1,160,0,0,2,1,2,0],
[53,1,0,123,282,0,1,95,1,2,1,2,3,0],
[63,0,0,108,269,0,1,169,1,1.8,1,2,2,0],
[54,1,0,110,206,0,0,108,1,0,1,1,2,0],
[66,1,0,112,212,0,0,132,1,0.1,2,1,2,0],
[55,0,0,180,327,0,2,117,1,3.4,1,0,2,0],
[49,1,2,118,149,0,0,126,0,0.8,2,3,2,0],
[54,1,0,122,286,0,0,116,1,3.2,1,2,2,0],
[56,1,0,130,283,1,0,103,1,1.6,0,0,3,0],
[46,1,0,120,249,0,0,144,0,0.8,2,0,3,0],
[61,1,3,134,234,0,1,145,0,2.6,1,2,2,0],
[67,1,0,120,237,0,1,71,0,1,1,0,2,0],
[58,1,0,100,234,0,1,156,0,0.1,2,1,3,0],
[47,1,0,110,275,0,0,118,1,1,1,1,2,0],
[52,1,0,125,212,0,1,168,0,1,2,2,3,0],
[58,1,0,146,218,0,1,105,0,2,1,1,3,0],
[57,1,1,124,261,0,1,141,0,0.3,2,0,3,0],
[58,0,1,136,319,1,0,152,0,0,2,2,2,0],
[61,1,0,138,166,0,0,125,1,3.6,1,1,2,0],
[42,1,0,136,315,0,1,125,1,1.8,1,0,1,0],
[52,1,0,128,204,1,1,156,1,1,1,0,0,0],
[59,1,2,126,218,1,1,134,0,2.2,1,1,1,0],
[40,1,0,152,223,0,1,181,0,0,2,0,3,0],
[61,1,0,140,207,0,0,138,1,1.9,2,1,3,0],
[46,1,0,140,311,0,1,120,1,1.8,1,2,3,0],
[59,1,3,134,204,0,1,162,0,0.8,2,2,2,0],
[57,1,1,154,232,0,0,164,0,0,2,1,2,0],
[57,1,0,110,335,0,1,143,1,3,1,1,3,0],
[55,0,0,128,205,0,2,130,1,2,1,1,3,0],
[61,1,0,148,203,0,1,161,0,0,2,1,3,0],
[58,1,0,114,318,0,2,140,0,4.4,0,3,1,0],
[58,0,0,170,225,1,0,146,1,2.8,1,2,1,0],
[67,1,2,152,212,0,0,150,0,0.8,1,0,3,0],
[44,1,0,120,169,0,1,144,1,2.8,0,0,1,0],
[63,1,0,140,187,0,0,144,1,4,2,2,3,0],
[63,0,0,124,197,0,1,136,1,0,1,0,2,0],
[59,1,0,164,176,1,0,90,0,1,1,2,1,0],
[57,0,0,140,241,0,1,1223,1,0.2,1,0,3,0],
[45,1,3,110,2264,0,1,132,0,1.2,1,0,3,0],
[68,1,0,144,193,21,1,141,0,3.4,1,2,3,0],
[57,1,0,130,131,0,31,115,1,1.2,1,1,3,0],
[57,0,1,130,2362,0,0,1274,0,0,1,1,2,0],
[57,0,1,130,236,10,0,1742227,0,1,1,2,0],
[44,1,220,120,1269,0,1,1444,1,2.8,0,0,61,0],
[63,1,02,140,187,0,0,144,1,4,2,2,3,0],
[63,0,0,1224,197,0,21,136,21,0,1,0,2,0],
[59,1,0,164,176,1,20,90,0,1,1,2,1,0],
[57,0,0,140,241,0,13,123,1,0.2,1,0,3,0],
[45,1,3,110,2264,0,13,132,0,12.2,1,0,3,0],
[68,1,0,144,193,1,1,1412,0,3.4,1,2,3,0]]
    df = pd.read_csv(data , columns = ['age','sex','cp',  'trestbps',  'chol',  'fbs',  'restecg',  'thalach',  'exang',  'oldpeak',  'slope',  'ca',  'thal'])

    X = df[['age','sex','cp',  'trestbps',  'chol',  'fbs',  'restecg',  'thalach',  'exang',  'oldpeak',  'slope',  'ca',  'thal']]
    #print(X)
    y = df['target']
    #print(y)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
    X_train = X
    y_train = y
    #nn_model = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1, random_state=0)
    nn_model = SVC(  probability = True)
    #nn_model = LogisticRegression()
    nn_model.fit(X_train, y_train)
    pred = nn_model.predict([list_data])
    #accuracy = nn_model.predict_log_proba([list_data])
    accuracy = nn_model.predict_proba([list_data])
    print(pred ,  accuracy)
    #print("Neural Network Accuracy: {:.2f}%".format(nn_model.score(X_test, y_test) * 100))
    #print("Prdicted Value is : ", format(pred))
    dataframe = str(df.head())
    #return (nn_model.score(X_test, y_test) * 100),(pred)
    if accuracy[0][0] <= 0.5:
        pred = "1"
        return (abs((accuracy[0][1])*100)) , (pred)
    else:
        pred = "0" 
        return (abs(accuracy[0][0])*100) , pred
    #return accuracy[0][0] , pred
    
    

@login_required(login_url="login")
def add_doctor(request,pid=None):
    doctor = None
    if pid:
        doctor = Doctor.objects.get(id=pid)
    if request.method == "POST":
        form = DoctorForm(request.POST, request.FILES, instance = doctor)
        if form.is_valid():
            new_doc = form.save()
            new_doc.status = 1
            if not pid:
                user = User.objects.create_user(password=request.POST['password'], username=request.POST['username'], first_name=request.POST['first_name'], last_name=request.POST['last_name'])
                new_doc.user = user
            new_doc.save()
            return redirect('view_doctor')
    d = {"doctor": doctor}
    return render(request, 'add_doctor.html', d)

@login_required(login_url="login")
def add_heartdetail(request):
    if request.method == "POST":
        # list_data = [57, 0, 1, 130, 236, 0, 0, 174, 0, 0.0, 1, 1, 2]
        list_data = []
        value_dict = eval(str(request.POST)[12:-1])
        print(value_dict)
        count = 0
        for key,value in value_dict.items():
            if count == 0:
                count = 1
                continue
            if key == "Gender" and (value[0] == "Male" or value[0] == 'male'):
                list_data.append(0)
                continue
            elif key == "Gender" and (value[0] == "female" or value[0] == 'Female'):
                list_data.append(1)
                continue
            list_data.append(value[0])

        # list_data = [57, 0, 1, 130, 236, 0, 0, 174, 0, 0.0, 1, 1, 2]
        accuracy,pred = prdict_heart_disease(list_data)
        patient = Patient.objects.get(user=request.user)
        Search_Data.objects.create(patient=patient, prediction_accuracy=accuracy, result=pred[0], values_list=list_data)
        rem = int(pred[0])
        print("Result = ",rem)
        if pred[0] == 0:
            pred = "<span style='color:green'>You are healthy</span>"
        else:
            pred = "<span style='color:red'>You are Unhealthy, Need to Checkup.</span>"
        return redirect('predict_desease', str(rem), str(accuracy))
    return render(request, 'add_heartdetail.html')

@login_required(login_url="login")
def predict_desease(request, pred, accuracy):
    doctor = Doctor.objects.filter(address__icontains=Patient.objects.get(user=request.user).address)
    d = {'pred': pred, 'accuracy':accuracy, 'doctor':doctor}
    return render(request, 'predict_disease.html', d)

@login_required(login_url="login")
def view_search_pat(request):
    doc = None
    try:
        doc = Doctor.objects.get(user=request.user)
        data = Search_Data.objects.filter(patient__address__icontains=doc.address).order_by('-id')
        #print(data)
    except:
        try:
            doc = Patient.objects.get(user=request.user)
            data = Search_Data.objects.filter(patient=doc).order_by('-id')
            print(data)
        except:
            data = Search_Data.objects.all().order_by('-id')
    return render(request,'view_search_pat.html',{'data':data})

@login_required(login_url="login")
def delete_doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    doc.delete()
    return redirect('view_doctor')

@login_required(login_url="login")
def delete_feedback(request,pid):
    doc = Feedback.objects.get(id=pid)
    doc.delete()
    return redirect('view_feedback')

@login_required(login_url="login")
def delete_patient(request,pid):
    doc = Patient.objects.get(id=pid)
    doc.delete()
    return redirect('view_patient')

@login_required(login_url="login")
def delete_searched(request,pid):
    doc = Search_Data.objects.get(id=pid)
    doc.delete()
    return redirect('view_search_pat')

@login_required(login_url="login")
def View_Doctor(request):
    doc = Doctor.objects.all()
    d = {'doc':doc}
    return render(request,'view_doctor.html',d)

@login_required(login_url="login")
def View_Patient(request):
    patient = Patient.objects.all()
    d = {'patient':patient}
    return render(request,'view_patient.html',d)

@login_required(login_url="login")
def View_Feedback(request):
    dis = Feedback.objects.all()
    d = {'dis':dis}
    return render(request,'view_feedback.html',d)

@login_required(login_url="login")
def View_My_Detail(request):
    terror = ""
    user = User.objects.get(id=request.user.id)
    error = ""
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    d = {'error': error,'pro':sign}
    return render(request,'profile_doctor.html',d)

@login_required(login_url="login")
def Edit_Doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    error = ""
    # type = Type.objects.all()
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        cat = request.POST['type']
        try:
            im = request.FILES['image']
            doc.image=im
            doc.save()
        except:
            pass
        dat = datetime.date.today()
        doc.user.first_name = f
        doc.user.last_name = l
        doc.user.email = e
        doc.contact = con
        doc.category = cat
        doc.address = add
        doc.user.save()
        doc.save()
        error = "create"
    d = {'error':error,'doc':doc,'type':type}
    return render(request,'edit_doctor.html',d)

@login_required(login_url="login")
def Edit_My_deatail(request):
    terror = ""
    print("Hii welvome")
    user = User.objects.get(id=request.user.id)
    error = ""
    # type = Type.objects.all()
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        try:
            im = request.FILES['image']
            sign.image = im
            sign.save()
        except:
            pass
        to1 = datetime.date.today()
        sign.user.first_name = f
        sign.user.last_name = l
        sign.user.email = e
        sign.contact = con
        if error != "pat":
            cat = request.POST['type']
            sign.category = cat
            sign.save()
        sign.address = add
        sign.user.save()
        sign.save()
        terror = "create"
    d = {'error':error,'terror':terror,'doc':sign}
    return render(request,'edit_profile.html',d)

@login_required(login_url='login')
def sent_feedback(request):
    terror = None
    if request.method == "POST":
        username = request.POST['uname']
        message = request.POST['msg']
        username = User.objects.get(username=username)
        Feedback.objects.create(user=username, messages=message)
        terror = "create"
    return render(request, 'sent_feedback.html',{'terror':terror})
