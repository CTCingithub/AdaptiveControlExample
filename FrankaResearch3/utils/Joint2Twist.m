function OUTPUT = Joint2Twist(JOINT,LOCATION)
    %JOINT2TWIST 此处显示有关此函数的摘要
    %   此处显示详细说明
    OUTPUT=[JOINT;Vector2Matrix_3x3(LOCATION)*JOINT];
end

