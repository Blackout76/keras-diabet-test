"��
XDeviceIDLE"IDLE(1�����ƠB9�����ƠBA�����ƠBI�����ƠBQ      �?Y      �?�Unknown
VHostIDLE"IDLE(133333e�@9��Q�nm@A33333e�@I��Q�nm@a,���D��?i,���D��?�Unknown
fHost_FusedMatMul"dense_1/Relu(1������-@9������-@A������-@I������-@a��16�?ip�P҉��?�Unknown
xHostMatMul"$gradients/dense_2/MatMul_grad/MatMul(1      +@9      +@A      +@I      +@a�	�F�?i��t��?�Unknown
fHost_FusedMatMul"dense_2/Relu(1333333)@9333333)@A333333)@I333333)@a���n��?i�0#\�?�Unknown
fHostGreaterEqual"GreaterEqual(1      '@9      '@A      '@I      '@a�����n}?iMk� ��?�Unknown
zHostMatMul"&gradients/dense_2/MatMul_grad/MatMul_1(1������%@9������%@A������%@I������%@a��t�{?i��T�H��?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1333333#@9333333@A333333#@I333333@a���.�x?i6沔l��?�Unknown
i	Host_FusedMatMul"dense_3/BiasAdd(1333333"@9333333"@A333333"@I333333"@aU��IJw?i3��&.�?�Unknown
x
HostMatMul"$gradients/dense_3/MatMul_grad/MatMul(1ffffff @9ffffff @Affffff @Iffffff @a��rݛ�t?i��^�W�?�Unknown
zHostMatMul"&gradients/dense_1/MatMul_grad/MatMul_1(1ffffff@9ffffff@Affffff@Iffffff@a��I�K�p?iR���y�?�Unknown
�Host_UnaryOpsComposition",metrics/accuracy/Round/unary_ops_composition(1������@9������@A������@I������@a�k�P3n?i�"AF���?�Unknown
VHostCast"Cast(1ffffff@9ffffff@Affffff@Iffffff@a���2�l?i�B�x���?�Unknown
zHostReadVariableOp"dense_3/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a5��:�i?i��e���?�Unknown
�HostTile"Kgradients/loss/dense_3_loss/binary_crossentropy/weighted_loss/Sum_grad/Tile(1������@9������@A������@I������@a5��:�i?i�zBQ���?�Unknown
zHostMatMul"&gradients/dense_3/MatMul_grad/MatMul_1(1333333@9333333@A333333@I333333@a���.�h?i�qq3\��?�Unknown
�HostSelect":loss/dense_3_loss/binary_crossentropy/logistic_loss/Select(1ffffff@9ffffff@Affffff@Iffffff@a���΋g?i#���?�Unknown
�HostBiasAddGrad"*gradients/dense_3/BiasAdd_grad/BiasAddGrad(1333333@9333333@A333333@I333333@a�"���f?iF�}��,�?�Unknown
�HostDynamicStitch"Ggradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/DynamicStitch(1ffffff@9ffffff@Affffff@Iffffff@a��rݛ�d?i N[M�A�?�Unknown
WHostMul"mul_11(1      @9      @A      @I      @ae��ёyd?i�-�`V�?�Unknown
�HostNeg"7loss/dense_3_loss/binary_crossentropy/logistic_loss/Neg(1333333@9333333@A333333@I333333@aܐ(Ƈ�c?ifD�fWj�?�Unknown
mHostReadVariableOp"ReadVariableOp_30(1ffffff@9ffffff@Affffff@Iffffff@aSl��}sc?i�ǭ��}�?�Unknown
�HostBiasAddGrad"*gradients/dense_2/BiasAdd_grad/BiasAddGrad(1������@9������@A������@I������@aA#9�imb?i� QN8��?�Unknown
jHostReadVariableOp"ReadVariableOp(1      @9      �?A      @I      �?a����_�a?i���"��?�Unknown
�HostMul"7loss/dense_3_loss/binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a����_�a?i�(���?�Unknown
eHostSum"metrics/accuracy/Sum(1      @9      @A      @I      @a����_�a?i�m���?�Unknown
�HostSelect"Tgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_1_grad/Select_1(1������@9������@A������@I������@a)���n�_?i_������?�Unknown
WHostMul"mul_13(1������@9������@A������@I������@a)���n�_?i̻�۳��?�Unknown
WHostMul"mul_28(1      @9      @A      @I      @a���Z�^?iG	��?�Unknown
gHostMean"metrics/accuracy/Mean(1333333@9333333@A333333@I333333@aGj�F�]?i8˘,��?�Unknown
THostPow"Pow(1ffffff@9ffffff@Affffff@Iffffff@a���2�\?i7��E<�?�Unknown
` HostAddN"gradients/AddN(1ffffff@9ffffff@Affffff@Iffffff@a���2�\?i6�$_� �?�Unknown
�!HostBiasAddGrad"*gradients/dense_1/BiasAdd_grad/BiasAddGrad(1������@9������@A������@I������@a��t�[?iV_nc.�?�Unknown
W"HostSub"sub_12(1������@9������@A������@I������@a��t�[?i���}5<�?�Unknown
z#HostReluGrad"$gradients/dense_2/Relu_grad/ReluGrad(1������@9������@A������@I������@a�k�]
�Z?i��Ȃ�I�?�Unknown
q$HostAssignVariableOp"AssignVariableOp_13(1      @9      @A      @I      @a�"AF��Y?i1��}PV�?�Unknown
p%HostAssignVariableOp"AssignVariableOp_9(1      @9      @A      @I      @a�"AF��Y?i��yc�?�Unknown
�&Host	ZerosLike"Tgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_grad/zeros_like(1      @9      @A      @I      @a�"AF��Y?iS�1t�o�?�Unknown
�'HostAssignAddVariableOp"&metrics/accuracy/AssignAddVariableOp_1(1333333@9333333@A333333@I333333@a���.�X?i�cIe1|�?�Unknown
�(HostSum"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/sub_grad/Sum(1ffffff@9ffffff@Affffff@Iffffff@a���΋W?i:UL���?�Unknown
V)HostAddV2"add(1������@9������@A������@I������@a�Gb ��V?i,kU):��?�Unknown
�*HostSelect"<loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_1(1������@9������@A������@I������@a�Gb ��V?iP�U}��?�Unknown
p+HostReadVariableOp"mul_6/ReadVariableOp(1������@9������@A������@I������@a�Gb ��V?it�U㿩�?�Unknown
p,HostAssignVariableOp"AssignVariableOp_6(1������ @9������ @A������ @I������ @aw��U?isYJ���?�Unknown
X-HostAddV2"add_8(1������ @9������ @A������ @I������ @aw��U?ir�>�?��?�Unknown
�.HostSelect"Pgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_grad/Select(1������ @9������ @A������ @I������ @aw��U?iqq3\���?�Unknown
�/Host	ZerosLike"Vgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_1_grad/zeros_like(1       @9       @A       @I       @ae��ёyT?iLX%<��?�Unknown
V0HostMul"mul_8(1       @9       @A       @I       @ae��ёyT?i'?�x��?�Unknown
q1HostAssignVariableOp"AssignVariableOp_11(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSl��}sS?i݀�2��?�Unknown
\2HostSquare"Square_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSl��}sS?i�¿k���?�Unknown
\3HostSquare"Square_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSl��}sS?iI�*���?�Unknown
l4HostMinimum"clip_by_value_1/Minimum(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSl��}sS?i�Ez�_�?�Unknown
l5HostMinimum"clip_by_value_4/Minimum(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSl��}sS?i��W��?�Unknown
�6HostGreaterEqual"@loss/dense_3_loss/binary_crossentropy/logistic_loss/GreaterEqual(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSl��}sS?ik�4g��?�Unknown
�7HostLog1p"9loss/dense_3_loss/binary_crossentropy/logistic_loss/Log1p(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSl��}sS?i!&�"�?�Unknown
W8HostMul"mul_20(1ffffff�?9ffffff�?Affffff�?Iffffff�?aSl��}sS?i�L��F,�?�Unknown
Z9HostSquare"Square(1�������?9�������?A�������?I�������?aA#9�imR?ii���}5�?�Unknown
X:HostAddV2"add_7(1�������?9�������?A�������?I�������?aA#9�imR?i���N�>�?�Unknown
g;HostCast"metrics/accuracy/Cast(1�������?9�������?A�������?I�������?aA#9�imR?i�"d�G�?�Unknown
W<HostMul"mul_17(1�������?9�������?A�������?I�������?aA#9�imR?i�5�!Q�?�Unknown
X=HostSqrt"Sqrt_3(1333333�?9333333�?A333333�?I333333�?a/��UgQ?i���b�Y�?�Unknown
z>HostReluGrad"$gradients/dense_1/Relu_grad/ReluGrad(1333333�?9333333�?A333333�?I333333�?a/��UgQ?i����b�?�Unknown
|?HostMean"*loss/dense_3_loss/binary_crossentropy/Mean(1333333�?9333333�?A333333�?I333333�?a/��UgQ?if���<k�?�Unknown
V@HostMul"mul_6(1333333�?9333333�?A333333�?I333333�?a/��UgQ?iӜMc�s�?�Unknown
qAHostAssignVariableOp"AssignVariableOp_16(1�������?9�������?A�������?I�������?a��tAaP?i�!|�?�Unknown
pBHostAssignVariableOp"AssignVariableOp_2(1�������?9�������?A�������?I�������?a��tAaP?ieA¤Q��?�Unknown
lCHostMinimum"clip_by_value_3/Minimum(1�������?9�������?A�������?I�������?a��tAaP?i��|E���?�Unknown
�DHostAddV2"Lgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Log1p_grad/add(1�������?9�������?A�������?I�������?a��tAaP?i��6沔�?�Unknown
pEHostAssignVariableOp"AssignVariableOp_7(1      �?9      �?A      �?I      �?a���Z�N?i��|`��?�Unknown
VFHostPow"Pow_1(1      �?9      �?A      �?I      �?a���Z�N?i?@���?�Unknown
\GHostSquare"Square_3(1      �?9      �?A      �?I      �?a���Z�N?ic�B����?�Unknown
YHHostAddV2"add_10(1      �?9      �?A      �?I      �?a���Z�N?i���@i��?�Unknown
�IHostCast">gradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/Cast(1      �?9      �?A      �?I      �?a���Z�N?i�G����?�Unknown
�JHostMaximum"Agradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/Maximum(1      �?9      �?A      �?I      �?a���Z�N?i��Nn���?�Unknown
�KHostFloorDiv"Bgradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/floordiv(1      �?9      �?A      �?I      �?a���Z�N?i��r��?�Unknown
iLHostEqual"metrics/accuracy/Equal(1      �?9      �?A      �?I      �?a���Z�N?iO����?�Unknown
mMHostRealDiv"metrics/accuracy/truediv(1      �?9      �?A      �?I      �?a���Z�N?i;�Z2���?�Unknown
WNHostMul"mul_15(1      �?9      �?A      �?I      �?a���Z�N?i_�	�z��?�Unknown
WOHostMul"mul_23(1      �?9      �?A      �?I      �?a���Z�N?i�V�_(��?�Unknown
WPHostMul"mul_25(1      �?9      �?A      �?I      �?a���Z�N?i�g����?�Unknown
qQHostReadVariableOp"mul_28/ReadVariableOp(1      �?9      �?A      �?I      �?a���Z�N?i˰����?�Unknown
VRHostMul"mul_3(1      �?9      �?A      �?I      �?a���Z�N?i�]�#1 �?�Unknown
WSHostSub"sub_10(1      �?9      �?A      �?I      �?a���Z�N?is���?�Unknown
^THostRealDiv"	truediv_1(1      �?9      �?A      �?I      �?a���Z�N?i7�!Q��?�Unknown
^UHostRealDiv"	truediv_3(1      �?9      �?A      �?I      �?a���Z�N?i[e��9�?�Unknown
nVHostAssignVariableOp"AssignVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�L?iZmstd�?�Unknown
pWHostAssignVariableOp"AssignVariableOp_8(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�L?iYu�%�?�Unknown
jXHostMinimum"clip_by_value/Minimum(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�L?iX}���,�?�Unknown
lYHostMinimum"clip_by_value_5/Minimum(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�L?iW�\�3�?�Unknown
�ZHostExp"7loss/dense_3_loss/binary_crossentropy/logistic_loss/Exp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�L?iV���;�?�Unknown
V[HostMul"mul_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�L?iU��39B�?�Unknown
V\HostMul"mul_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�L?iT�E�cI�?�Unknown
W]HostSub"sub_13(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�L?iS��L�P�?�Unknown
W^HostSub"sub_17(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�L?iR��ٸW�?�Unknown
q_HostAssignVariableOp"AssignVariableOp_10(1�������?9�������?A�������?I�������?a�k�]
�J?i-#\`^�?�Unknown
V`HostSqrt"Sqrt(1�������?9�������?A�������?I�������?a�k�]
�J?is��e�?�Unknown
XaHostSqrt"Sqrt_1(1�������?9�������?A�������?I�������?a�k�]
�J?i��Qa�k�?�Unknown
YbHostAddV2"add_11(1�������?9�������?A�������?I�������?a�k�]
�J?i�8��Vr�?�Unknown
YcHostAddV2"add_12(1�������?9�������?A�������?I�������?a�k�]
�J?i���f�x�?�Unknown
XdHostAddV2"add_2(1�������?9�������?A�������?I�������?a�k�]
�J?it���?�Unknown
zeHostReadVariableOp"dense_1/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�k�]
�J?iOa�kM��?�Unknown
�fHostRealDiv"Agradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/truediv(1�������?9�������?A�������?I�������?a�k�]
�J?i*�F���?�Unknown
�gHost
Reciprocal"Rgradients/loss/dense_3_loss/binary_crossentropy/weighted_loss/truediv_grad/RealDiv(1�������?9�������?A�������?I�������?a�k�]
�J?i'�p���?�Unknown
�hHostSum"7loss/dense_3_loss/binary_crossentropy/weighted_loss/Sum(1�������?9�������?A�������?I�������?a�k�]
�J?i��u�C��?�Unknown
�iHostMul"7loss/dense_3_loss/binary_crossentropy/weighted_loss/mul(1�������?9�������?A�������?I�������?a�k�]
�J?i��v��?�Unknown
VjHostMul"mul_4(1�������?9�������?A�������?I�������?a�k�]
�J?i�O�����?�Unknown
VkHostSub"sub_4(1�������?9�������?A�������?I�������?a�k�]
�J?iq�;{:��?�Unknown
XlHostAddV2"add_3(1333333�?9333333�?A333333�?I333333�?a���.�H?i'p��^��?�Unknown
dmHostMaximum"clip_by_value_1(1333333�?9333333�?A333333�?I333333�?a���.�H?i�-Sl���?�Unknown
dnHostMaximum"clip_by_value_4(1333333�?9333333�?A333333�?I333333�?a���.�H?i������?�Unknown
ioHostCast"metrics/accuracy/Cast_1(1333333�?9333333�?A333333�?I333333�?a���.�H?iI�j]���?�Unknown
TpHostMul"mul(1333333�?9333333�?A333333�?I333333�?a���.�H?i�f�����?�Unknown
WqHostMul"mul_16(1333333�?9333333�?A333333�?I333333�?a���.�H?i�$�N��?�Unknown
WrHostMul"mul_19(1333333�?9333333�?A333333�?I333333�?a���.�H?ik��9��?�Unknown
VsHostSub"sub_7(1333333�?9333333�?A333333�?I333333�?a���.�H?i!��?^��?�Unknown
\tHostRealDiv"truediv(1333333�?9333333�?A333333�?I333333�?a���.�H?i�]%����?�Unknown
^uHostRealDiv"	truediv_4(1333333�?9333333�?A333333�?I333333�?a���.�H?i��0���?�Unknown
pvHostAssignVariableOp"AssignVariableOp_4(1�������?9�������?A�������?I�������?a�Gb ��F?i41�H��?�Unknown
XwHostSqrt"Sqrt_4(1�������?9�������?A�������?I�������?a�Gb ��F?i�L����?�Unknown
bxHostMaximum"clip_by_value(1�������?9�������?A�������?I�������?a�Gb ��F?iCe1|���?�Unknown
dyHostMaximum"clip_by_value_5(1�������?9�������?A�������?I�������?a�Gb ��F?i�}��,�?�Unknown
�zHostSum"Hgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss_grad/Sum_1(1�������?9�������?A�������?I�������?a�Gb ��F?ig�1Y��?�Unknown
W{HostMul"mul_12(1�������?9�������?A�������?I�������?a�Gb ��F?i����o�?�Unknown
q|HostReadVariableOp"mul_16/ReadVariableOp(1�������?9�������?A�������?I�������?a�Gb ��F?i��16�?�Unknown
W}HostMul"mul_18(1�������?9�������?A�������?I�������?a�Gb ��F?i౤��?�Unknown
W~HostMul"mul_21(1�������?9�������?A�������?I�������?a�Gb ��F?i��1T�?�Unknown
WHostSub"sub_15(1�������?9�������?A�������?I�������?a�Gb ��F?iA���#�?�Unknown
X�HostSub"sub_18(1�������?9�������?A�������?I�������?a�Gb ��F?i�)2�)�?�Unknown
W�HostSub"sub_2(1�������?9�������?A�������?I�������?a�Gb ��F?ieB�^8/�?�Unknown
n�HostReadVariableOp"ReadVariableOp_15(1      �?9      �?A      �?I      �?ae��ёyD?iҵ&�V4�?�Unknown
m�HostReadVariableOp"ReadVariableOp_8(1      �?9      �?A      �?I      �?ae��ёyD?i?)�'u9�?�Unknown
]�HostSquare"Square_5(1      �?9      �?A      �?I      �?ae��ёyD?i����>�?�Unknown
m�HostMinimum"clip_by_value_2/Minimum(1      �?9      �?A      �?I      �?ae��ёyD?i��C�?�Unknown
��HostBroadcastTo"Egradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/BroadcastTo(1      �?9      �?A      �?I      �?ae��ёyD?i���T�H�?�Unknown
��HostSelect"Rgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_1_grad/Select(1      �?9      �?A      �?I      �?ae��ёyD?i��l��M�?�Unknown
��HostSum"Lgradients/loss/dense_3_loss/binary_crossentropy/weighted_loss/mul_grad/Sum_1(1      �?9      �?A      �?I      �?ae��ёyD?i`j�S�?�Unknown
X�HostMul"mul_10(1      �?9      �?A      �?I      �?ae��ёyD?i��U�+X�?�Unknown
X�HostMul"mul_14(1      �?9      �?A      �?I      �?ae��ёyD?i:Q��I]�?�Unknown
X�HostMul"mul_24(1      �?9      �?A      �?I      �?ae��ёyD?i��>Khb�?�Unknown
X�HostMul"mul_26(1      �?9      �?A      �?I      �?ae��ёyD?i8���g�?�Unknown
X�HostMul"mul_29(1      �?9      �?A      �?I      �?ae��ёyD?i��'�l�?�Unknown
X�HostMul"mul_30(1      �?9      �?A      �?I      �?ae��ёyD?i��x�q�?�Unknown
W�HostMul"mul_5(1      �?9      �?A      �?I      �?ae��ёyD?i[���v�?�Unknown
X�HostSub"sub_14(1      �?9      �?A      �?I      �?ae��ёyD?i��A |�?�Unknown
W�HostSub"sub_3(1      �?9      �?A      �?I      �?ae��ёyD?i5y����?�Unknown
r�HostAssignVariableOp"AssignVariableOp_12(1�������?9�������?A�������?I�������?aA#9�imB?i~Gb ���?�Unknown
r�HostAssignVariableOp"AssignVariableOp_14(1�������?9�������?A�������?I�������?aA#9�imB?i��ZU��?�Unknown
p�HostReadVariableOp"Cast/ReadVariableOp(1�������?9�������?A�������?I�������?aA#9�imB?i�3����?�Unknown
n�HostReadVariableOp"ReadVariableOp_25(1�������?9�������?A�������?I�������?aA#9�imB?iY�����?�Unknown
n�HostReadVariableOp"ReadVariableOp_27(1�������?9�������?A�������?I�������?aA#9�imB?i��j'��?�Unknown
Y�HostSqrt"Sqrt_2(1�������?9�������?A�������?I�������?aA#9�imB?i�Nn��?�Unknown
]�HostSquare"Square_1(1�������?9�������?A�������?I�������?aA#9�imB?i4�^��?�Unknown
Z�HostAddV2"add_14(1�������?9�������?A�������?I�������?aA#9�imB?i}�?y���?�Unknown
Y�HostAddV2"add_9(1�������?9�������?A�������?I�������?aA#9�imB?iƹ�Ӕ��?�Unknown
e�HostMaximum"clip_by_value_3(1�������?9�������?A�������?I�������?aA#9�imB?i�.0��?�Unknown
m�HostMinimum"clip_by_value_6/Minimum(1�������?9�������?A�������?I�������?aA#9�imB?iXVz�˳�?�Unknown
z�HostReadVariableOp"dense_3/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?aA#9�imB?i�$��f��?�Unknown
��HostAdd"3loss/dense_3_loss/binary_crossentropy/logistic_loss(1�������?9�������?A�������?I�������?aA#9�imB?i��K=��?�Unknown
��HostSub"7loss/dense_3_loss/binary_crossentropy/logistic_loss/sub(1�������?9�������?A�������?I�������?aA#9�imB?i3������?�Unknown
��HostCast"Eloss/dense_3_loss/binary_crossentropy/weighted_loss/num_elements/Cast(1�������?9�������?A�������?I�������?aA#9�imB?i|��8��?�Unknown
��HostAssignAddVariableOp"$metrics/accuracy/AssignAddVariableOp(1�������?9�������?A�������?I�������?aA#9�imB?i�]�L���?�Unknown
q�HostReadVariableOp"mul_8/ReadVariableOp(1�������?9�������?A�������?I�������?aA#9�imB?i,�o��?�Unknown
U�HostSub"sub(1�������?9�������?A�������?I�������?aA#9�imB?iW�W��?�Unknown
X�HostSub"sub_11(1�������?9�������?A�������?I�������?aA#9�imB?i���[���?�Unknown
W�HostSub"sub_5(1�������?9�������?A�������?I�������?aA#9�imB?i�)�A��?�Unknown
W�HostSub"sub_8(1�������?9�������?A�������?I�������?aA#9�imB?i2e����?�Unknown
W�HostSub"sub_9(1�������?9�������?A�������?I�������?aA#9�imB?i{3�jx��?�Unknown
_�HostRealDiv"	truediv_2(1�������?9�������?A�������?I�������?aA#9�imB?i�d���?�Unknown
q�HostAssignVariableOp"AssignVariableOp_1(1�������?9�������?A�������?I�������?a��tAa@?i�*�,��?�Unknown
r�HostAssignVariableOp"AssignVariableOp_15(1�������?9�������?A�������?I�������?a��tAa@?iTfD��?�Unknown
q�HostAssignVariableOp"AssignVariableOp_3(1�������?9�������?A�������?I�������?a��tAa@?i0}{�\��?�Unknown
n�HostReadVariableOp"ReadVariableOp_12(1�������?9�������?A�������?I�������?a��tAa@?iT��u��?�Unknown
n�HostReadVariableOp"ReadVariableOp_17(1�������?9�������?A�������?I�������?a��tAa@?ix�5W���?�Unknown
n�HostReadVariableOp"ReadVariableOp_19(1�������?9�������?A�������?I�������?a��tAa@?i������?�Unknown
m�HostReadVariableOp"ReadVariableOp_2(1�������?9�������?A�������?I�������?a��tAa@?i�!����?�Unknown
n�HostReadVariableOp"ReadVariableOp_21(1�������?9�������?A�������?I�������?a��tAa@?i�JMH��?�Unknown
n�HostReadVariableOp"ReadVariableOp_26(1�������?9�������?A�������?I�������?a��tAa@?it����?�Unknown
m�HostReadVariableOp"ReadVariableOp_5(1�������?9�������?A�������?I�������?a��tAa@?i,���?�Unknown
Y�HostSqrt"Sqrt_5(1�������?9�������?A�������?I�������?a��tAa@?iP�d9�?�Unknown
Y�HostSqrt"Sqrt_6(1�������?9�������?A�������?I�������?a��tAa@?it���7�?�Unknown
Z�HostAddV2"add_15(1�������?9�������?A�������?I�������?a��tAa@?i��O �?�Unknown
��HostMul"Lgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Log1p_grad/mul(1�������?9�������?A�������?I�������?a��tAa@?i�A|*h$�?�Unknown
��HostNeg"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/sub_grad/Neg(1�������?9�������?A�������?I�������?a��tAa@?i�j�z�(�?�Unknown
q�HostReadVariableOp"mul_1/ReadVariableOp(1�������?9�������?A�������?I�������?a��tAa@?i�6˘,�?�Unknown
r�HostReadVariableOp"mul_21/ReadVariableOp(1�������?9�������?A�������?I�������?a��tAa@?i(���0�?�Unknown
r�HostReadVariableOp"mul_23/ReadVariableOp(1�������?9�������?A�������?I�������?a��tAa@?iL��k�4�?�Unknown
r�HostReadVariableOp"mul_26/ReadVariableOp(1�������?9�������?A�������?I�������?a��tAa@?ipN��8�?�Unknown
X�HostSub"sub_16(1�������?9�������?A�������?I�������?a��tAa@?i�8��<�?�Unknown
W�HostSub"sub_6(1�������?9�������?A�������?I�������?a��tAa@?i�a]A�?�Unknown
_�HostRealDiv"	truediv_5(1�������?9�������?A�������?I�������?a��tAa@?i܊e�*E�?�Unknown
r�HostAssignVariableOp"AssignVariableOp_17(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i���H�?�Unknown
q�HostAssignVariableOp"AssignVariableOp_5(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?iܒ:UL�?�Unknown
o�HostReadVariableOp"Pow/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�Z��O�?�Unknown
m�HostReadVariableOp"ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?iܚ��S�?�Unknown
n�HostReadVariableOp"ReadVariableOp_10(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i��W�?�Unknown
n�HostReadVariableOp"ReadVariableOp_11(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?iܢNS�Z�?�Unknown
n�HostReadVariableOp"ReadVariableOp_13(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�&��?^�?�Unknown
n�HostReadVariableOp"ReadVariableOp_18(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?iܪ���a�?�Unknown
n�HostReadVariableOp"ReadVariableOp_20(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�.C&je�?�Unknown
n�HostReadVariableOp"ReadVariableOp_22(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?iܲ�l�h�?�Unknown
m�HostReadVariableOp"ReadVariableOp_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�6沔l�?�Unknown
m�HostReadVariableOp"ReadVariableOp_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?iܺ7�)p�?�Unknown
m�HostReadVariableOp"ReadVariableOp_6(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�>�?�s�?�Unknown
m�HostReadVariableOp"ReadVariableOp_7(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i��څTw�?�Unknown
Y�HostAddV2"add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�F,��z�?�Unknown
Z�HostAddV2"add_13(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i��}~�?�Unknown
Z�HostAddV2"add_17(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�N�X��?�Unknown
Y�HostAddV2"add_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�� ����?�Unknown
e�HostMaximum"clip_by_value_6(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�Vr�>��?�Unknown
��HostSum"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/mul_grad/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i���+Ԍ�?�Unknown
��HostSum"Fgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss_grad/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�^ri��?�Unknown
��HostMul"Lgradients/loss/dense_3_loss/binary_crossentropy/weighted_loss/mul_grad/Mul_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i��f����?�Unknown
r�HostReadVariableOp"mul_11/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�f�����?�Unknown
r�HostReadVariableOp"mul_18/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i��	E)��?�Unknown
X�HostMul"mul_22(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�n[����?�Unknown
X�HostMul"mul_27(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i���S��?�Unknown
q�HostReadVariableOp"mul_3/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�v���?�Unknown
W�HostMul"mul_9(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i��O^~��?�Unknown
W�HostSub"sub_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i�~����?�Unknown
X�HostSub"sub_19(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i��ꨰ�?�Unknown
_�HostRealDiv"	truediv_6(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���2�<?i܆D1>��?�Unknown
q�HostReadVariableOp"Pow_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a���.�8?i�e�mP��?�Unknown
n�HostReadVariableOp"ReadVariableOp_14(1333333�?9333333�?A333333�?I333333�?a���.�8?i�DЩb��?�Unknown
n�HostReadVariableOp"ReadVariableOp_16(1333333�?9333333�?A333333�?I333333�?a���.�8?im#�t��?�Unknown
n�HostReadVariableOp"ReadVariableOp_23(1333333�?9333333�?A333333�?I333333�?a���.�8?iH\"���?�Unknown
n�HostReadVariableOp"ReadVariableOp_29(1333333�?9333333�?A333333�?I333333�?a���.�8?i#�^���?�Unknown
m�HostReadVariableOp"ReadVariableOp_9(1333333�?9333333�?A333333�?I333333�?a���.�8?i��皫��?�Unknown
Z�HostAddV2"add_18(1333333�?9333333�?A333333�?I333333�?a���.�8?iٞ-׽��?�Unknown
Y�HostAddV2"add_5(1333333�?9333333�?A333333�?I333333�?a���.�8?i�}s���?�Unknown
Y�HostAddV2"add_6(1333333�?9333333�?A333333�?I333333�?a���.�8?i�\�O���?�Unknown
e�HostMaximum"clip_by_value_2(1333333�?9333333�?A333333�?I333333�?a���.�8?ij;�����?�Unknown
z�HostReadVariableOp"dense_2/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a���.�8?iEE���?�Unknown
��HostMul"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Exp_grad/mul(1333333�?9333333�?A333333�?I333333�?a���.�8?i ����?�Unknown
��Host
Reciprocal"Sgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Log1p_grad/Reciprocal(1333333�?9333333�?A333333�?I333333�?a���.�8?i���@+��?�Unknown
��HostNeg"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Neg_grad/Neg(1333333�?9333333�?A333333�?I333333�?a���.�8?iֶ}=��?�Unknown
��HostMul"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/mul_grad/Mul(1333333�?9333333�?A333333�?I333333�?a���.�8?i��\�O��?�Unknown
��HostSum"Lgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/sub_grad/Sum_1(1333333�?9333333�?A333333�?I333333�?a���.�8?i�t��a��?�Unknown
��HostRealDiv";loss/dense_3_loss/binary_crossentropy/weighted_loss/truediv(1333333�?9333333�?A333333�?I333333�?a���.�8?igS�1t��?�Unknown
��HostReadVariableOp"'metrics/accuracy/truediv/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a���.�8?iB2.n���?�Unknown
W�HostMul"mul_7(1333333�?9333333�?A333333�?I333333�?a���.�8?it����?�Unknown
n�HostReadVariableOp"ReadVariableOp_24(1      �?9      �?A      �?I      �?ae��ёy4?i�J��'��?�Unknown
Z�HostAddV2"add_16(1      �?9      �?A      �?I      �?ae��ёy4?i������?�Unknown
z�HostReadVariableOp"dense_1/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?ae��ёy4?iB�"AF��?�Unknown
{�HostReadVariableOp"dense_2/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?ae��ёy4?i��\s���?�Unknown
|�HostReadVariableOp"metrics/accuracy/ReadVariableOp(1      �?9      �?A      �?I      �?ae��ёy4?i�1��d��?�Unknown
r�HostReadVariableOp"mul_13/ReadVariableOp(1      �?9      �?A      �?I      �?ae��ёy4?igk�����?�Unknown
n�HostReadVariableOp"ReadVariableOp_28(1�������?9�������?A�������?I�������?a��tAa0?i�������?�Unknown*��
fHost_FusedMatMul"dense_1/Relu(1������-@9������-@A������-@I������-@a(X����?i(X����?�Unknown
xHostMatMul"$gradients/dense_2/MatMul_grad/MatMul(1      +@9      +@A      +@I      +@a�z�)V�?i~��S/�?�Unknown
fHost_FusedMatMul"dense_2/Relu(1333333)@9333333)@A333333)@I333333)@a����~�?iz�d�θ?�Unknown
fHostGreaterEqual"GreaterEqual(1      '@9      '@A      '@I      '@aڡKr)՛?i��n�ÿ?�Unknown
zHostMatMul"&gradients/dense_2/MatMul_grad/MatMul_1(1������%@9������%@A������%@I������%@a��0�t#�?i��A�k&�?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1333333#@9333333@A333333#@I333333@a=���;�?i�%����?�Unknown
iHost_FusedMatMul"dense_3/BiasAdd(1333333"@9333333"@A333333"@I333333"@aJ=]�+�?iz�d���?�Unknown
xHostMatMul"$gradients/dense_3/MatMul_grad/MatMul(1ffffff @9ffffff @Affffff @Iffffff @a�׃یؓ?ilHխ�I�?�Unknown
z	HostMatMul"&gradients/dense_1/MatMul_grad/MatMul_1(1ffffff@9ffffff@Affffff@Iffffff@a)[r�?i�:���H�?�Unknown
�
Host_UnaryOpsComposition",metrics/accuracy/Round/unary_ops_composition(1������@9������@A������@I������@aln�"	��?i�ѵe��?�Unknown
VHostCast"Cast(1ffffff@9ffffff@Affffff@Iffffff@aHխ�I�?i�W� �a�?�Unknown
zHostReadVariableOp"dense_3/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a�x�෇?iq4��?�Unknown
�HostTile"Kgradients/loss/dense_3_loss/binary_crossentropy/weighted_loss/Sum_grad/Tile(1������@9������@A������@I������@a�x�෇?iF�	E��?�Unknown
zHostMatMul"&gradients/dense_3/MatMul_grad/MatMul_1(1333333@9333333@A333333@I333333@a=���;�?iإ�$��?�Unknown
�HostSelect":loss/dense_3_loss/binary_crossentropy/logistic_loss/Select(1ffffff@9ffffff@Affffff@Iffffff@a{�<(!D�?i�_�EI�?�Unknown
�HostBiasAddGrad"*gradients/dense_3/BiasAdd_grad/BiasAddGrad(1333333@9333333@A333333@I333333@aW� �aЄ?i'������?�Unknown
�HostDynamicStitch"Ggradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/DynamicStitch(1ffffff@9ffffff@Affffff@Iffffff@a�׃ی؃?i�s8���?�Unknown
WHostMul"mul_11(1      @9      @A      @I      @a3O�e�\�?i^ڡKr)�?�Unknown
�HostNeg"7loss/dense_3_loss/binary_crossentropy/logistic_loss/Neg(1333333@9333333@A333333@I333333@a�����?i�"x��?�Unknown
mHostReadVariableOp"ReadVariableOp_30(1ffffff@9ffffff@Affffff@Iffffff@aq>Hz�d�?i�R�v�S�?�Unknown
�HostBiasAddGrad"*gradients/dense_2/BiasAdd_grad/BiasAddGrad(1������@9������@A������@I������@a�-ˎ�l�?i��j;��?�Unknown
jHostReadVariableOp"ReadVariableOp(1      @9      �?A      @I      �?aM��?i 3��f�?�Unknown
�HostMul"7loss/dense_3_loss/binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aM��?iJv���?�Unknown
eHostSum"metrics/accuracy/Sum(1      @9      @A      @I      @aM��?it�Í�u�?�Unknown
�HostSelect"Tgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_1_grad/Select_1(1������@9������@A������@I������@a�%��~?i�oԯ���?�Unknown
WHostMul"mul_13(1������@9������@A������@I������@a�%��~?i��ѵe�?�Unknown
WHostMul"mul_28(1      @9      @A      @I      @a�����
}?i��G����?�Unknown
gHostMean"metrics/accuracy/Mean(1333333@9333333@A333333@I333333@a
�*�|?i#O�.J�?�Unknown
THostPow"Pow(1ffffff@9ffffff@Affffff@Iffffff@aHխ�I{?ixB���?�Unknown
`HostAddN"gradients/AddN(1ffffff@9ffffff@Affffff@Iffffff@aHխ�I{?iͽ	i#�?�Unknown
�HostBiasAddGrad"*gradients/dense_1/BiasAdd_grad/BiasAddGrad(1������@9������@A������@I������@a��0�t#z?i߀b<���?�Unknown
W HostSub"sub_12(1������@9������@A������@I������@a��0�t#z?i�C�$��?�Unknown
z!HostReluGrad"$gradients/dense_2/Relu_grad/ReluGrad(1������@9������@A������@I������@aó��+y?i�f��X�?�Unknown
q"HostAssignVariableOp"AssignVariableOp_13(1      @9      @A      @I      @a �6��3x?iL�b����?�Unknown
p#HostAssignVariableOp"AssignVariableOp_9(1      @9      @A      @I      @a �6��3x?i��_�p�?�Unknown
�$Host	ZerosLike"Tgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_grad/zeros_like(1      @9      @A      @I      @a �6��3x?id�\@{�?�Unknown
�%HostAssignAddVariableOp"&metrics/accuracy/AssignAddVariableOp_1(1333333@9333333@A333333@I333333@a=���;w?i����/��?�Unknown
�&HostSum"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/sub_grad/Sum(1ffffff@9ffffff@Affffff@Iffffff@a{�<(!Dv?i�zLp@1�?�Unknown
V'HostAddV2"add(1������@9������@A������@I������@a�p�<LLu?ivx?�q��?�Unknown
�(HostSelect"<loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_1(1������@9������@A������@I������@a�p�<LLu?i9v2Ң��?�Unknown
p)HostReadVariableOp"mul_6/ReadVariableOp(1������@9������@A������@I������@a�p�<LLu?i�s%�0�?�Unknown
p*HostAssignVariableOp"AssignVariableOp_6(1������ @9������ @A������ @I������ @a�_BQwTt?i|}j�%��?�Unknown
X+HostAddV2"add_8(1������ @9������ @A������ @I������ @a�_BQwTt?i����w��?�Unknown
�,HostSelect"Pgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_grad/Select(1������ @9������ @A������ @I������ @a�_BQwTt?i>Hz�d�?�Unknown
�-Host	ZerosLike"Vgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_1_grad/zeros_like(1       @9       @A       @I       @a3O�e�\s?i��E9�?�Unknown
V.HostMul"mul_8(1       @9       @A       @I       @a3O�e�\s?iz]W�_�?�Unknown
q/HostAssignVariableOp"AssignVariableOp_11(1ffffff�?9ffffff�?Affffff�?Iffffff�?aq>Hz�dr?i����?�Unknown
\0HostSquare"Square_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?aq>Hz�dr?it~��j��?�Unknown
\1HostSquare"Square_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?aq>Hz�dr?i��'4��?�Unknown
l2HostMinimum"clip_by_value_1/Minimum(1ffffff�?9ffffff�?Affffff�?Iffffff�?aq>Hz�dr?in������?�Unknown
l3HostMinimum"clip_by_value_4/Minimum(1ffffff�?9ffffff�?Affffff�?Iffffff�?aq>Hz�dr?i�/�]��?�Unknown
�4HostGreaterEqual"@loss/dense_3_loss/binary_crossentropy/logistic_loss/GreaterEqual(1ffffff�?9ffffff�?Affffff�?Iffffff�?aq>Hz�dr?ih����<�?�Unknown
�5HostLog1p"9loss/dense_3_loss/binary_crossentropy/logistic_loss/Log1p(1ffffff�?9ffffff�?Affffff�?Iffffff�?aq>Hz�dr?i�P��Za�?�Unknown
W6HostMul"mul_20(1ffffff�?9ffffff�?Affffff�?Iffffff�?aq>Hz�dr?ib�.$��?�Unknown
Z7HostSquare"Square(1�������?9�������?A�������?I�������?a�-ˎ�lq?i�w����?�Unknown
X8HostAddV2"add_7(1�������?9�������?A�������?I�������?a�-ˎ�lq?i����?�Unknown
g9HostCast"metrics/accuracy/Cast(1�������?9�������?A�������?I�������?a�-ˎ�lq?is����?�Unknown
W:HostMul"mul_17(1�������?9�������?A�������?I�������?a�-ˎ�lq?i�:,��?�Unknown
X;HostSqrt"Sqrt_3(1333333�?9333333�?A333333�?I333333�?a�N�#up?i�r:v2�?�Unknown
z<HostReluGrad"$gradients/dense_1/Relu_grad/ReluGrad(1333333�?9333333�?A333333�?I333333�?a�N�#up?iBs��`S�?�Unknown
|=HostMean"*loss/dense_3_loss/binary_crossentropy/Mean(1333333�?9333333�?A333333�?I333333�?a�N�#up?i| �Jt�?�Unknown
V>HostMul"mul_6(1333333�?9333333�?A333333�?I333333�?a�N�#up?i��F5��?�Unknown
q?HostAssignVariableOp"AssignVariableOp_16(1�������?9�������?A�������?I�������?aS�o��n?i�M��/��?�Unknown
p@HostAssignVariableOp"AssignVariableOp_2(1�������?9�������?A�������?I�������?aS�o��n?i��%K*��?�Unknown
lAHostMinimum"clip_by_value_3/Minimum(1�������?9�������?A�������?I�������?aS�o��n?i����$��?�Unknown
�BHostAddV2"Lgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Log1p_grad/add(1�������?9�������?A�������?I�������?aS�o��n?i4��?�Unknown
pCHostAssignVariableOp"AssignVariableOp_7(1      �?9      �?A      �?I      �?a�����
m?iܝy*.�?�Unknown
VDHostPow"Pow_1(1      �?9      �?A      �?I      �?a�����
m?i�6m5K�?�Unknown
\EHostSquare"Square_3(1      �?9      �?A      �?I      �?a�����
m?i�+�`@h�?�Unknown
YFHostAddV2"add_10(1      �?9      �?A      �?I      �?a�����
m?i��gTK��?�Unknown
�GHostCast">gradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/Cast(1      �?9      �?A      �?I      �?a�����
m?i�{ HV��?�Unknown
�HHostMaximum"Agradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/Maximum(1      �?9      �?A      �?I      �?a�����
m?i�#�;a��?�Unknown
�IHostFloorDiv"Bgradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/floordiv(1      �?9      �?A      �?I      �?a�����
m?i��1/l��?�Unknown
iJHostEqual"metrics/accuracy/Equal(1      �?9      �?A      �?I      �?a�����
m?i�s�"w��?�Unknown
mKHostRealDiv"metrics/accuracy/truediv(1      �?9      �?A      �?I      �?a�����
m?i�c��?�Unknown
WLHostMul"mul_15(1      �?9      �?A      �?I      �?a�����
m?i���	�3�?�Unknown
WMHostMul"mul_23(1      �?9      �?A      �?I      �?a�����
m?i�k���P�?�Unknown
WNHostMul"mul_25(1      �?9      �?A      �?I      �?a�����
m?i�-�m�?�Unknown
qOHostReadVariableOp"mul_28/ReadVariableOp(1      �?9      �?A      �?I      �?a�����
m?i���䭊�?�Unknown
VPHostMul"mul_3(1      �?9      �?A      �?I      �?a�����
m?i�c^ظ��?�Unknown
WQHostSub"sub_10(1      �?9      �?A      �?I      �?a�����
m?i������?�Unknown
^RHostRealDiv"	truediv_1(1      �?9      �?A      �?I      �?a�����
m?i�������?�Unknown
^SHostRealDiv"	truediv_3(1      �?9      �?A      �?I      �?a�����
m?i}[(����?�Unknown
nTHostAssignVariableOp"AssignVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�Ik?iR	����?�Unknown
pUHostAssignVariableOp"AssignVariableOp_8(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�Ik?i'��F5�?�Unknown
jVHostMinimum"clip_by_value/Minimum(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�Ik?i�dm�+P�?�Unknown
lWHostMinimum"clip_by_value_5/Minimum(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�Ik?i�/�Fk�?�Unknown
�XHostExp"7loss/dense_3_loss/binary_crossentropy/logistic_loss/Exp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�Ik?i���#b��?�Unknown
VYHostMul"mul_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�Ik?i{n�m}��?�Unknown
VZHostMul"mul_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�Ik?iPt����?�Unknown
W[HostSub"sub_13(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�Ik?i%�5���?�Unknown
W\HostSub"sub_17(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�Ik?i�w�J���?�Unknown
q]HostAssignVariableOp"AssignVariableOp_10(1�������?9�������?A�������?I�������?aó��+i?i�+����?�Unknown
V^HostSqrt"Sqrt(1�������?9�������?A�������?I�������?aó��+i?ib�̊&%�?�Unknown
X_HostSqrt"Sqrt_1(1�������?9�������?A�������?I�������?aó��+i?i��*R>�?�Unknown
Y`HostAddV2"add_11(1�������?9�������?A�������?I�������?aó��+i?i�F��}W�?�Unknown
YaHostAddV2"add_12(1�������?9�������?A�������?I�������?aó��+i?i~��j�p�?�Unknown
XbHostAddV2"add_2(1�������?9�������?A�������?I�������?aó��+i?i2�w
Չ�?�Unknown
zcHostReadVariableOp"dense_1/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?aó��+i?i�ab� ��?�Unknown
�dHostRealDiv"Agradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/truediv(1�������?9�������?A�������?I�������?aó��+i?i�MJ,��?�Unknown
�eHost
Reciprocal"Rgradients/loss/dense_3_loss/binary_crossentropy/weighted_loss/truediv_grad/RealDiv(1�������?9�������?A�������?I�������?aó��+i?iN�7�W��?�Unknown
�fHostSum"7loss/dense_3_loss/binary_crossentropy/weighted_loss/Sum(1�������?9�������?A�������?I�������?aó��+i?i}"����?�Unknown
�gHostMul"7loss/dense_3_loss/binary_crossentropy/weighted_loss/mul(1�������?9�������?A�������?I�������?aó��+i?i�0*��?�Unknown
VhHostMul"mul_4(1�������?9�������?A�������?I�������?aó��+i?ij���� �?�Unknown
ViHostSub"sub_4(1�������?9�������?A�������?I�������?aó��+i?i��i:�?�Unknown
XjHostAddV2"add_3(1333333�?9333333�?A333333�?I333333�?a=���;g?i�Q�_BQ�?�Unknown
dkHostMaximum"clip_by_value_1(1333333�?9333333�?A333333�?I333333�?a=���;g?iB
V~h�?�Unknown
dlHostMaximum"clip_by_value_4(1333333�?9333333�?A333333�?I333333�?a=���;g?i��L��?�Unknown
imHostCast"metrics/accuracy/Cast_1(1333333�?9333333�?A333333�?I333333�?a=���;g?if~1B���?�Unknown
TnHostMul"mul(1333333�?9333333�?A333333�?I333333�?a=���;g?i�7E82��?�Unknown
WoHostMul"mul_16(1333333�?9333333�?A333333�?I333333�?a=���;g?i��X.n��?�Unknown
WpHostMul"mul_19(1333333�?9333333�?A333333�?I333333�?a=���;g?i�l$���?�Unknown
VqHostSub"sub_7(1333333�?9333333�?A333333�?I333333�?a=���;g?i�d����?�Unknown
\rHostRealDiv"truediv(1333333�?9333333�?A333333�?I333333�?a=���;g?i@�"�?�Unknown
^sHostRealDiv"	truediv_4(1333333�?9333333�?A333333�?I333333�?a=���;g?i�ק^"�?�Unknown
ptHostAssignVariableOp"AssignVariableOp_4(1�������?9�������?A�������?I�������?a�p�<LLe?iC��R�7�?�Unknown
XuHostSqrt"Sqrt_4(1�������?9�������?A�������?I�������?a�p�<LLe?i�V!��L�?�Unknown
bvHostMaximum"clip_by_value(1�������?9�������?A�������?I�������?a�p�<LLe?i%^�Bb�?�Unknown
dwHostMaximum"clip_by_value_5(1�������?9�������?A�������?I�������?a�p�<LLe?i�՚7�w�?�Unknown
�xHostSum"Hgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss_grad/Sum_1(1�������?9�������?A�������?I�������?a�p�<LLe?i�׃ی�?�Unknown
WyHostMul"mul_12(1�������?9�������?A�������?I�������?a�p�<LLe?ixT�'��?�Unknown
qzHostReadVariableOp"mul_16/ReadVariableOp(1�������?9�������?A�������?I�������?a�p�<LLe?i�Qt��?�Unknown
W{HostMul"mul_18(1�������?9�������?A�������?I�������?a�p�<LLe?iZӍh���?�Unknown
W|HostMul"mul_21(1�������?9�������?A�������?I�������?a�p�<LLe?i˒ʴ��?�Unknown
W}HostSub"sub_15(1�������?9�������?A�������?I�������?a�p�<LLe?i<RY��?�Unknown
W~HostSub"sub_18(1�������?9�������?A�������?I�������?a�p�<LLe?i�DM��?�Unknown
VHostSub"sub_2(1�������?9�������?A�������?I�������?a�p�<LLe?iр��!�?�Unknown
n�HostReadVariableOp"ReadVariableOp_15(1      �?9      �?A      �?I      �?a3O�e�\c?im��;N5�?�Unknown
m�HostReadVariableOp"ReadVariableOp_8(1      �?9      �?A      �?I      �?a3O�e�\c?i�[LުH�?�Unknown
]�HostSquare"Square_5(1      �?9      �?A      �?I      �?a3O�e�\c?i!��\�?�Unknown
m�HostMinimum"clip_by_value_2/Minimum(1      �?9      �?A      �?I      �?a3O�e�\c?iZ�#do�?�Unknown
��HostBroadcastTo"Egradients/loss/dense_3_loss/binary_crossentropy/Mean_grad/BroadcastTo(1      �?9      �?A      �?I      �?a3O�e�\c?i��}����?�Unknown
��HostSelect"Rgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Select_1_grad/Select(1      �?9      �?A      �?I      �?a3O�e�\c?i�p�g��?�Unknown
��HostSum"Lgradients/loss/dense_3_loss/binary_crossentropy/weighted_loss/mul_grad/Sum_1(1      �?9      �?A      �?I      �?a3O�e�\c?iG6I
z��?�Unknown
X�HostMul"mul_10(1      �?9      �?A      �?I      �?a3O�e�\c?i����ּ�?�Unknown
X�HostMul"mul_14(1      �?9      �?A      �?I      �?a3O�e�\c?i��O3��?�Unknown
X�HostMul"mul_24(1      �?9      �?A      �?I      �?a3O�e�\c?i4�z���?�Unknown
X�HostMul"mul_26(1      �?9      �?A      �?I      �?a3O�e�\c?i�K�����?�Unknown
X�HostMul"mul_29(1      �?9      �?A      �?I      �?a3O�e�\c?i�F6I
�?�Unknown
X�HostMul"mul_30(1      �?9      �?A      �?I      �?a3O�e�\c?i!֫إ�?�Unknown
W�HostMul"mul_5(1      �?9      �?A      �?I      �?a3O�e�\c?ip�{1�?�Unknown
X�HostSub"sub_14(1      �?9      �?A      �?I      �?a3O�e�\c?i�`w_D�?�Unknown
W�HostSub"sub_3(1      �?9      �?A      �?I      �?a3O�e�\c?i&ݿ�W�?�Unknown
r�HostAssignVariableOp"AssignVariableOp_12(1�������?9�������?A�������?I�������?a�-ˎ�la?i<�k�(i�?�Unknown
r�HostAssignVariableOp"AssignVariableOp_14(1�������?9�������?A�������?I�������?a�-ˎ�la?ij����z�?�Unknown
p�HostReadVariableOp"Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�-ˎ�la?i������?�Unknown
n�HostReadVariableOp"ReadVariableOp_25(1�������?9�������?A�������?I�������?a�-ˎ�la?i�R�o��?�Unknown
n�HostReadVariableOp"ReadVariableOp_27(1�������?9�������?A�������?I�������?a�-ˎ�la?i���ܮ�?�Unknown
Y�HostSqrt"Sqrt_2(1�������?9�������?A�������?I�������?a�-ˎ�la?i"�5�I��?�Unknown
]�HostSquare"Square_1(1�������?9�������?A�������?I�������?a�-ˎ�la?iP�ċ���?�Unknown
Z�HostAddV2"add_14(1�������?9�������?A�������?I�������?a�-ˎ�la?i~S�#��?�Unknown
Y�HostAddV2"add_9(1�������?9�������?A�������?I�������?a�-ˎ�la?i�J�|���?�Unknown
e�HostMaximum"clip_by_value_3(1�������?9�������?A�������?I�������?a�-ˎ�la?i�qu��?�Unknown
m�HostMinimum"clip_by_value_6/Minimum(1�������?9�������?A�������?I�������?a�-ˎ�la?i��mj�?�Unknown
z�HostReadVariableOp"dense_3/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a�-ˎ�la?i6��f�(�?�Unknown
��HostAdd"3loss/dense_3_loss/binary_crossentropy/logistic_loss(1�������?9�������?A�������?I�������?a�-ˎ�la?idw_D:�?�Unknown
��HostSub"7loss/dense_3_loss/binary_crossentropy/logistic_loss/sub(1�������?9�������?A�������?I�������?a�-ˎ�la?i�B�W�K�?�Unknown
��HostCast"Eloss/dense_3_loss/binary_crossentropy/weighted_loss/num_elements/Cast(1�������?9�������?A�������?I�������?a�-ˎ�la?i�;P]�?�Unknown
��HostAssignAddVariableOp"$metrics/accuracy/AssignAddVariableOp(1�������?9�������?A�������?I�������?a�-ˎ�la?i���H�n�?�Unknown
q�HostReadVariableOp"mul_8/ReadVariableOp(1�������?9�������?A�������?I�������?a�-ˎ�la?i�XA��?�Unknown
U�HostSub"sub(1�������?9�������?A�������?I�������?a�-ˎ�la?iJo�9e��?�Unknown
X�HostSub"sub_11(1�������?9�������?A�������?I�������?a�-ˎ�la?ix:v2Ң�?�Unknown
W�HostSub"sub_5(1�������?9�������?A�������?I�������?a�-ˎ�la?i�+?��?�Unknown
W�HostSub"sub_8(1�������?9�������?A�������?I�������?a�-ˎ�la?i�Г#���?�Unknown
W�HostSub"sub_9(1�������?9�������?A�������?I�������?a�-ˎ�la?i�"��?�Unknown
_�HostRealDiv"	truediv_2(1�������?9�������?A�������?I�������?a�-ˎ�la?i0g����?�Unknown
q�HostAssignVariableOp"AssignVariableOp_1(1�������?9�������?A�������?I�������?aS�o��^?i<8ic��?�Unknown
r�HostAssignVariableOp"AssignVariableOp_15(1�������?9�������?A�������?I�������?aS�o��^?iH	!���?�Unknown
q�HostAssignVariableOp"AssignVariableOp_3(1�������?9�������?A�������?I�������?aS�o��^?iT�� ��?�Unknown
n�HostReadVariableOp"ReadVariableOp_12(1�������?9�������?A�������?I�������?aS�o��^?i`��O{&�?�Unknown
n�HostReadVariableOp"ReadVariableOp_17(1�������?9�������?A�������?I�������?aS�o��^?il|H��5�?�Unknown
n�HostReadVariableOp"ReadVariableOp_19(1�������?9�������?A�������?I�������?aS�o��^?ixM �uE�?�Unknown
m�HostReadVariableOp"ReadVariableOp_2(1�������?9�������?A�������?I�������?aS�o��^?i��;�T�?�Unknown
n�HostReadVariableOp"ReadVariableOp_21(1�������?9�������?A�������?I�������?aS�o��^?i��o�pd�?�Unknown
n�HostReadVariableOp"ReadVariableOp_26(1�������?9�������?A�������?I�������?aS�o��^?i��'��s�?�Unknown
m�HostReadVariableOp"ReadVariableOp_5(1�������?9�������?A�������?I�������?aS�o��^?i���'k��?�Unknown
Y�HostSqrt"Sqrt_5(1�������?9�������?A�������?I�������?aS�o��^?i�b�v��?�Unknown
Y�HostSqrt"Sqrt_6(1�������?9�������?A�������?I�������?aS�o��^?i�3O�e��?�Unknown
Z�HostAddV2"add_15(1�������?9�������?A�������?I�������?aS�o��^?i���?�Unknown
��HostMul"Lgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Log1p_grad/mul(1�������?9�������?A�������?I�������?aS�o��^?i�վb`��?�Unknown
��HostNeg"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/sub_grad/Neg(1�������?9�������?A�������?I�������?aS�o��^?i�v����?�Unknown
q�HostReadVariableOp"mul_1/ReadVariableOp(1�������?9�������?A�������?I�������?aS�o��^?i�w. [��?�Unknown
r�HostReadVariableOp"mul_21/ReadVariableOp(1�������?9�������?A�������?I�������?aS�o��^?i�H�N���?�Unknown
r�HostReadVariableOp"mul_23/ReadVariableOp(1�������?9�������?A�������?I�������?aS�o��^?i��U��?�Unknown
r�HostReadVariableOp"mul_26/ReadVariableOp(1�������?9�������?A�������?I�������?aS�o��^?i�U���?�Unknown
X�HostSub"sub_16(1�������?9�������?A�������?I�������?aS�o��^?i �;P�?�Unknown
W�HostSub"sub_6(1�������?9�������?A�������?I�������?aS�o��^?i,�ŉ�-�?�Unknown
_�HostRealDiv"	truediv_5(1�������?9�������?A�������?I�������?aS�o��^?i8^}�J=�?�Unknown
r�HostAssignVariableOp"AssignVariableOp_17(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i#5^}�J�?�Unknown
q�HostAssignVariableOp"AssignVariableOp_5(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i?"fX�?�Unknown
o�HostReadVariableOp"Pow/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i����e�?�Unknown
m�HostReadVariableOp"ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i� l�s�?�Unknown
n�HostReadVariableOp"ReadVariableOp_10(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?iϐ���?�Unknown
n�HostReadVariableOp"ReadVariableOp_11(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i�gµ���?�Unknown
n�HostReadVariableOp"ReadVariableOp_13(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i�>�Z*��?�Unknown
n�HostReadVariableOp"ReadVariableOp_18(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i������?�Unknown
n�HostReadVariableOp"ReadVariableOp_20(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i{�d�E��?�Unknown
n�HostReadVariableOp"ReadVariableOp_22(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?if�EI���?�Unknown
m�HostReadVariableOp"ReadVariableOp_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?iQ�&�`��?�Unknown
m�HostReadVariableOp"ReadVariableOp_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i<q����?�Unknown
m�HostReadVariableOp"ReadVariableOp_6(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i'H�7|��?�Unknown
m�HostReadVariableOp"ReadVariableOp_7(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i��	��?�Unknown
Y�HostAddV2"add_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i������?�Unknown
Z�HostAddV2"add_13(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i�̊&%�?�Unknown
Z�HostAddV2"add_17(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?iӣk˲#�?�Unknown
Y�HostAddV2"add_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i�zLp@1�?�Unknown
e�HostMaximum"clip_by_value_6(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i�Q-�>�?�Unknown
��HostSum"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/mul_grad/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i�(�[L�?�Unknown
��HostSum"Fgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss_grad/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i��^�Y�?�Unknown
��HostMul"Lgradients/loss/dense_3_loss/binary_crossentropy/weighted_loss/mul_grad/Mul_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?ij��wg�?�Unknown
r�HostReadVariableOp"mul_11/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?iU���u�?�Unknown
r�HostReadVariableOp"mul_18/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i@��M���?�Unknown
X�HostMul"mul_22(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i+[r���?�Unknown
X�HostMul"mul_27(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i2S����?�Unknown
q�HostReadVariableOp"mul_3/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i	4<;��?�Unknown
W�HostMul"mul_9(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i���ȸ�?�Unknown
W�HostSub"sub_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i׶��V��?�Unknown
X�HostSub"sub_19(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i�*���?�Unknown
_�HostRealDiv"	truediv_6(1ffffff�?9ffffff�?Affffff�?Iffffff�?aHխ�I[?i�d��q��?�Unknown
q�HostReadVariableOp"Pow_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a=���;W?ivA����?�Unknown
n�HostReadVariableOp"ReadVariableOp_14(1333333�?9333333�?A333333�?I333333�?a=���;W?i?�ŭ��?�Unknown
n�HostReadVariableOp"ReadVariableOp_16(1333333�?9333333�?A333333�?I333333�?a=���;W?i���K�?�Unknown
n�HostReadVariableOp"ReadVariableOp_23(1333333�?9333333�?A333333�?I333333�?a=���;W?i��޻��?�Unknown
n�HostReadVariableOp"ReadVariableOp_29(1333333�?9333333�?A333333�?I333333�?a=���;W?i��趇�?�Unknown
m�HostReadVariableOp"ReadVariableOp_9(1333333�?9333333�?A333333�?I333333�?a=���;W?ic��%'�?�Unknown
Z�HostAddV2"add_18(1333333�?9333333�?A333333�?I333333�?a=���;W?i,n���2�?�Unknown
Y�HostAddV2"add_5(1333333�?9333333�?A333333�?I333333�?a=���;W?i�J�a>�?�Unknown
Y�HostAddV2"add_6(1333333�?9333333�?A333333�?I333333�?a=���;W?i�'��I�?�Unknown
e�HostMaximum"clip_by_value_2(1333333�?9333333�?A333333�?I333333�?a=���;W?i���U�?�Unknown
z�HostReadVariableOp"dense_2/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a=���;W?iP�#�;a�?�Unknown
��HostMul"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Exp_grad/mul(1333333�?9333333�?A333333�?I333333�?a=���;W?i�-��l�?�Unknown
��Host
Reciprocal"Sgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Log1p_grad/Reciprocal(1333333�?9333333�?A333333�?I333333�?a=���;W?i�7�wx�?�Unknown
��HostNeg"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/Neg_grad/Neg(1333333�?9333333�?A333333�?I333333�?a=���;W?i�wA���?�Unknown
��HostMul"Jgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/mul_grad/Mul(1333333�?9333333�?A333333�?I333333�?a=���;W?itTK����?�Unknown
��HostSum"Lgradients/loss/dense_3_loss/binary_crossentropy/logistic_loss/sub_grad/Sum_1(1333333�?9333333�?A333333�?I333333�?a=���;W?i=1U�Q��?�Unknown
��HostRealDiv";loss/dense_3_loss/binary_crossentropy/weighted_loss/truediv(1333333�?9333333�?A333333�?I333333�?a=���;W?i_{��?�Unknown
��HostReadVariableOp"'metrics/accuracy/truediv/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a=���;W?i��hv���?�Unknown
W�HostMul"mul_7(1333333�?9333333�?A333333�?I333333�?a=���;W?i��rq+��?�Unknown
n�HostReadVariableOp"ReadVariableOp_24(1      �?9      �?A      �?I      �?a3O�e�\S?i@������?�Unknown
Z�HostAddV2"add_16(1      �?9      �?A      �?I      �?a3O�e�\S?i�����?�Unknown
z�HostReadVariableOp"dense_1/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a3O�e�\S?i�oe6��?�Unknown
{�HostReadVariableOp"dense_2/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a3O�e�\S?i8R>����?�Unknown
|�HostReadVariableOp"metrics/accuracy/ReadVariableOp(1      �?9      �?A      �?I      �?a3O�e�\S?i�4q���?�Unknown
r�HostReadVariableOp"mul_13/ReadVariableOp(1      �?9      �?A      �?I      �?a3O�e�\S?i��XA��?�Unknown
n�HostReadVariableOp"ReadVariableOp_28(1�������?9�������?A�������?I�������?aS�o��N?i     �?�Unknown