// NB: This shader is very different from the DCTL implementation in the same repo
//     as it contains hacks to improve the rendering of red hues in SDR, cyan hues
//     in HDR and blue hues in both. The node graph in DaVinci Resolve looks like this:
//
//     SDR:
//
//     //-------\\       //----------------------------------------------------------\\
//     ||       ||------>|| Old ZCAM DRT v9 with better SDR reds but otherwise worse ||
//     || Input ||       ||     Hue Qualifier: Center 35, Width 14.2, Soft, 20.2     ||-----------------X
//     ||       ||\      \\----------------------------------------------------------//                 |
//     \\-------// \                                                                                    |
//                  \--->//-------------------------------------------------------------------\\        |
//                       || New ZCAM DRT v11 with better rendering overall but worse SDR reds ||        |
//                       ||             Inverted hue qualifier from the above one             ||------X |
//                       \\-------------------------------------------------------------------//      | |
//                                                                                                    v v
//                                                                         //--------\\            //------\\
//                                                                         || Output ||<-----------|| Lerp ||
//                                                                         \\--------//            \\------// 
//
//     HDR:
//     Only ZCAM DRT v11 node, parameters tweaked to have more desaturation, use P3D65 limiting primaries,
//     and to output to HDR10.
//
// Since these specific hacks were made specifically for compatibility with Baldur's Gate 3 old assets,
// this shader is currently for GDC advisors eyes only. It needs to be cleaned up in order to be used in
// a slide deck.

// Complex code makes fxc confuse functions and variables and throw the
// uninitialized variable warning spuriously.
#pragma warning(disable: 4000)

cbuffer CB_Globals : register (b0)
{
    float4 Params[7];

    // Alternate SSTS parameters for red hues in SDR
    float4 HACK_SDR_AlternateHighCoeffs;
    float4 HACK_SDR_AlternateTsPoints;          // x: min x, y: mid x, z: max x, w: max y
}

RWTexture3D<unorm float4> LUT : register(u0);

//#include "Shaders/HLSL/ACES.shdh"

#define PI 3.14159265358979f

static const row_major float3x3 Bt709_2_XYZ_MAT =
{
	0.412390917540f, 0.357584357262f, 0.180480793118f, 
	0.212639078498f, 0.715168714523f, 0.072192311287f, 
	0.019330825657f, 0.119194783270f, 0.950532138348f
};

static const row_major float3x3 XYZ_2_Bt709_MAT =
{
	 3.240969072272f, -1.537382761751f, -0.498610656716f,
	-0.969243658180f,  1.875967438967f,  0.041555079772f,
	 0.055630080818f, -0.203976958118f,  1.056971528280f
};

static const row_major float3x3 P3D65_2_XYZ_MAT =
{
	0.486570895f, 0.2656676470f, 0.1982172580f,
	0.228974566f, 0.6917384270f, 0.0792869031f,
	0.000000000f, 0.0451133773f, 1.0439442400f
};

static const row_major float3x3 XYZ_2_P3D65_MAT =
{
	 2.49349691f, -0.93138362f, -0.40271078f,  
	-0.82948897f,  1.76266406f,  0.02362469f, 
	 0.03584583f, -0.07617239f,  0.95688452f
};

static const row_major float3x3 Bt2020_2_XYZ_MAT =
{
	0.636958122253f, 0.144616916776f, 0.168880969286f, 
	0.262700229883f, 0.677998125553f, 0.059301715344f, 
	0.000000000000f, 0.028072696179f, 1.060985088348f
};

static const row_major float3x3 XYZ_2_Bt2020_MAT =
{
	 1.716650983017f, -0.355670745884f, -0.253366234419f,
	-0.666684264434f,  1.616481102428f,  0.015768536970f,
	 0.017639856590f, -0.042770613240f,  0.942103094219f
};

static const row_major float3x3 XYZ_2_LMS_CAT02_MAT = 
{ 
	 0.7328f, 0.4296f, -0.1624f,
	-0.7036f, 1.6975f,  0.0061f,
	 0.0030f, 0.0136f,  0.9834f
};

static const row_major float3x3 LMS_CAT02_2_XYZ_MAT = 
{
	 1.09612382f, -0.27886900f, 0.18274518f,
	 0.45436904f,  0.47353315f, 0.07209780f,
	-0.00962761f, -0.00569803f, 1.01532564f
};

// NB: This is not the official XYZ to ZCAM matrix. It has been modified a bit to prevent blues from turning cyan.
static const row_major float3x3 XYZ_2_LMS_ZCAM_MAT = 
{
	 0.4048027990f, 0.579999030f, 0.0246349014f,
	-0.2015099970f, 1.142175910f, 0.0315739214f,
	-0.0166008007f, 0.264800012f, 0.6684799190f
};

static const row_major float3x3 LMS_ZCAM_2_XYZ_MAT =
{
	 1.9734058400f, -0.996146977f, -0.0256737620f,
	 0.3506458400f,  0.708214223f, -0.0463727117f,
	-0.0898918733f, -0.305277616f,  1.5136629300f
};

static const row_major float3x3 LMS_ZCAM_2_Izazbz_MAT =
{
	0.000000f,  1.000000f,  0.000000f, 
	3.524000f, -4.066708f,  0.542708f, 
	0.199076f,  1.096799f, -1.295875f
};

static const row_major float3x3 Izazbz_2_LMS_ZCAM_MAT =
{
	1.00000000f,  0.2772100570f,  0.116094626f,
	1.00000000f, -0.0000000000f,  0.000000000f,
	1.00000000f,  0.0425858013f, -0.753844559f
};

static const row_major float3x3 AP0_2_XYZ_MAT =
{
    0.9525523959f, 0.0000000000f,  0.0000936786f,
    0.3439664498f, 0.7281660966f, -0.0721325464f,
    0.0000000000f, 0.0000000000f,  1.0088251844f
};

static const row_major float3x3 XYZ_2_AP0_MAT =
{
     1.0498110175f, 0.0000000000f, -0.0000974845f,
    -0.4959030231f, 1.3733130458f,  0.0982400361f,
     0.0000000000f, 0.0000000000f,  0.9912520182f
};

static const row_major float3x3 AP1_2_AP0_MAT =
{
     0.6954522414f,  0.1406786965f,  0.1638690622f,
     0.0447945634f,  0.8596711185f,  0.0955343182f,
    -0.0055258826f,  0.0040252103f,  1.0015006723f
};

static const float3 ACES_ReferenceWhite =
{
	0.9526460745698463f, 
	1.0f, 
	1.0088251843515859f
};

static const float3 D65_ReferenceWhite = 
{
	0.95045592705167148f, 
	1.0f, 
	1.0890577507598784f
};

// Base functions from SMPTE ST 2084-2014

// Constants from SMPTE ST 2084-2014
static const float pq_m1 = 0.1593017578125; // ( 2610.0 / 4096.0 ) / 4.0;
static const float pq_m2 = 78.84375; // ( 2523.0 / 4096.0 ) * 128.0;
static const float pq_c1 = 0.8359375; // 3424.0 / 4096.0 or pq_c3 - pq_c2 + 1.0;
static const float pq_c2 = 18.8515625; // ( 2413.0 / 4096.0 ) * 32.0;
static const float pq_c3 = 18.6875; // ( 2392.0 / 4096.0 ) * 32.0;

static const float pq_C = 10000.0;

// Converts from the non-linear perceptually quantized space to linear cd/m^2
// Note that this is in float, and assumes normalization from 0 - 1
// (0 - pq_C for linear) and does not handle the integer coding in the Annex 
// sections of SMPTE ST 2084-2014
float3 ST2084_2_Y(in float3 N)
{
  // Note that this does NOT handle any of the signal range
  // considerations from 2084 - this assumes full range (0 - 1)
  const float3 Np = (N == 0.0f) ? 0.0f : pow(abs(N), 1.0 / pq_m2);
  const float3 L = max(Np - pq_c1, 0.0) / (pq_c2 - pq_c3 * Np);
  return (L == 0.0f) ? 0.0f : pow( abs(L), 1.0 / pq_m1 ) * pq_C; // returns cd/m^2
}

// Converts from linear cd/m^2 to the non-linear perceptually quantized space
// Note that this is in float, and assumes normalization from 0 - 1
// (0 - pq_C for linear) and does not handle the integer coding in the Annex 
// sections of SMPTE ST 2084-2014
float3 Y_2_ST2084(in float3 C)
{
	// Note that this does NOT handle any of the signal range
	// considerations from 2084 - this returns full range (0 - 1)
	const float3 L = C / pq_C;
	const float3 Lm = (L == 0.0f) ? 0.0f : pow(abs(L), pq_m1);
	const float3 N = (pq_c1 + pq_c2 * Lm) / (1.0 + pq_c3 * Lm);
	return pow(abs(N), pq_m2);
}

// "Moving Frostbite to Physically Based Rendering 2.0"
// Sebastien Lagarde (Electronic Arts Frostbite) & Charles de Rousiers (Electronic Arts Frostbite)
// SIGGRAPH 2014
float3 sRGB_2_Y(in float3 sRGBCol)
{
	float3 linearRGBLo = sRGBCol / 12.92f;
	float3 linearRGBHi = pow(abs((sRGBCol + 0.055f)) / 1.055f, 2.4f);
	float3 linearRGB = (sRGBCol <= 0.04045f) ? linearRGBLo : linearRGBHi;
	return linearRGB;
}

float3 Y_2_sRGB(in float3 linearCol)
{
	float3 sRGBLo = linearCol * 12.92f;
	float3 sRGBHi = (pow(abs(linearCol), 1.0f/2.4f) * 1.055f) - 0.055f;
	float3 sRGB = (linearCol <= 0.0031308f) ? sRGBLo : sRGBHi;
	return sRGB;
}

float Linear_2_AcesCc(float lin)
{
	lin = max(lin, 0.f); // AcesCc is undefined if linear is below 0
	lin = lin <= 0.000030517578125 ? lin * 0.5f + 0.0000152587890625f : lin;
	
	return (log2(lin) + 9.72f) / 17.52f;
}

float AcesCc_2_Linear(float AcesCc)
{
	float x = exp2(AcesCc * 17.52f - 9.72f);
	return AcesCc <= -0.301369863f ? (x - 0.0000152587890625f) * 2.0f : x;
}

float3 Linear_2_AcesCc(float3 lin)
{
	lin = max(lin, 0.f); // AcesCc is undefined if linear is below 0
	lin = lin <= 0.000030517578125 ? lin * 0.5f + 0.0000152587890625f : lin;
	
	return (log2(lin) + 9.72f) / 17.52f;
}

float3 AcesCc_2_Linear(float3 AcesCc)
{
	float3 x = exp2(AcesCc * 17.52f - 9.72f);
	return AcesCc <= -0.301369863f ? (x - 0.0000152587890625f) * 2.0f : x;
}

float3 hsv_to_rgb(in const float3 c)
{
    const float4 K = float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
    const float3 p = abs(frac(c.xxx + K.xyz) * 6.0f - K.www);
    return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}

float3 rgb_to_hsv(in const float3 c)
{
    const float4 K = float4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f);
    const float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    const float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
    const float d = q.x - min(q.w, q.y);
    const float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

struct TsParams
{
    float2 Min;
    float2 Mid;
    float2 Max;
    float SlopeLow;
    float SlopeHigh;
    float CoefsLow[8]; // Should be only 6 coefficients, but there's a driver issue with Vulkan on AMD
    float CoefsHigh[8]; // Should be only 6 coefficients, but there's a driver issue with Vulkan on AMD
    float MinLum;
	float MidLum;
    float MaxLum;
    float HDRPaperWhite;
    float HDRGamma;
    int OutputDevice;
};

#define HALF_POS_INF 65504.0f

#define TS_PARAMS_MIN_X     Params[0].x //Params[0]
#define TS_PARAMS_MIN_Y     Params[0].y //Params[1]
#define TS_PARAMS_MIN_SLOPE Params[0].z //Params[2]

#define TS_PARAMS_MID_X     Params[0].w //Params[3]
#define TS_PARAMS_MID_Y     Params[1].x //Params[4]

// mid slope is unused for forward ssts computation and we don't want wasted space
#define TS_PARAMS_HDR_PAPER_WHITE Params[1].y //Params[5]

#define TS_PARAMS_MAX_X     Params[1].z //Params[6]
#define TS_PARAMS_MAX_Y     Params[1].w //Params[7]
#define TS_PARAMS_MAX_SLOPE Params[2].x //Params[8]

#define TS_PARAMS_COEFS_LOW_START_INDEX  9   // 9,  10, 11, 12, 13, 14
#define TS_PARAMS_COEFS_LOW_0 Params[2].y
#define TS_PARAMS_COEFS_LOW_1 Params[2].z
#define TS_PARAMS_COEFS_LOW_2 Params[2].w
#define TS_PARAMS_COEFS_LOW_3 Params[3].x
#define TS_PARAMS_COEFS_LOW_4 Params[3].y
#define TS_PARAMS_COEFS_LOW_5 Params[3].z

#define TS_PARAMS_COEFS_HIGH_START_INDEX 15  // 15, 16, 17, 18, 19, 20
#define TS_PARAMS_COEFS_HIGH_0 Params[3].w
#define TS_PARAMS_COEFS_HIGH_1 Params[4].x
#define TS_PARAMS_COEFS_HIGH_2 Params[4].y
#define TS_PARAMS_COEFS_HIGH_3 Params[4].z
#define TS_PARAMS_COEFS_HIGH_4 Params[4].w
#define TS_PARAMS_COEFS_HIGH_5 Params[5].x

#define TS_PARAMS_MIN_LUM      Params[5].y //Params[21]
#define TS_PARAMS_MID_LUM      Params[5].z //Params[22]
#define TS_PARAMS_MAX_LUM      Params[5].w //Params[23]

#define TS_PARAMS_LIMIT_HIGH   Params[6].x //Params[24]

#define TS_PARAMS_HDR_GAMMA Params[6].y
#define TS_PARAMS_OUTPUT_DEVICE asuint(Params[6].z) //asuint(Params[26])

#define HALF_MIN 0.0000000596046448f

static const float gmCuspMidBlend = 0.5f;
static const float gmFocusDistance = 0.5f;
static const float gmThresh = 0.75f;
static const float gmLimit = 1.2f;
static const float gmPower = 1.2f;
static const float gmSmoothCusps = 0.2f; // If need be, we can raise this for HDR to reduce saturation.

float3x3 GetLimitingPrimaries_XYZ_2_RGB_Matrix(const in int OutputDevice)
{
    return (OutputDevice == 0) ? XYZ_2_Bt709_MAT : XYZ_2_P3D65_MAT;
}

float3x3 GetLimitingPrimaries_RGB_2_XYZ_Matrix(const in int OutputDevice)
{
    return (OutputDevice == 0) ? Bt709_2_XYZ_MAT : P3D65_2_XYZ_MAT;
}

float3x3 GetOutputPrimaries_XYZ_2_RGB_Matrix(const in int OutputDevice)
{
    return (OutputDevice == 0) ? XYZ_2_Bt709_MAT : XYZ_2_Bt2020_MAT;
}

float3 CAT_Zhai2018(const in float3 XYZ_b, const in float3 XYZ_wb, const in float3 XYZ_wd, const in float D_b, const in float D_d)
{
    float3 XYZ_wo = float3(100.0f, 100.0f, 100.0f);
    float3 RGB_b = mul(XYZ_2_LMS_CAT02_MAT, XYZ_b);
    float3 RGB_wb = mul(XYZ_2_LMS_CAT02_MAT, XYZ_wb);
    float3 RGB_wd = mul(XYZ_2_LMS_CAT02_MAT, XYZ_wd);
    float3 RGB_wo = mul(XYZ_2_LMS_CAT02_MAT, XYZ_wo);

    float3 D_RGB_b = D_b * (XYZ_wb.y / XYZ_wo.y) * (RGB_wo / RGB_wb) + 1.0f - D_b;
    float3 D_RGB_d = D_d * (XYZ_wd.y / XYZ_wo.y) * (RGB_wo / RGB_wd) + 1.0f - D_d;
    float3 D_RGB = D_RGB_b / D_RGB_d;

    float3 RGB_d = D_RGB * RGB_b;
    float3 XYZ_d = mul(LMS_CAT02_2_XYZ_MAT, RGB_d);

    return XYZ_d;
}

// NB: spow, ZCAM_PQ_to_Linear and Linear_to_ZCAM_PQ handle powers with a negative base in a weird way
//     in order to preserve invalid negative inputs for intermediate calculations.
float spow(const in float x, const in float y)
{
    return (x == 0.0f) ? 0.0f : sign(x) * pow(abs(x), y);
}

float ZCAM_PQ_to_Linear(const in float pq)
{
    float x = pow(abs(pq), 1.0f / 134.034375f);
    x = min(max(x, 0.0f), 1.0087f); // Remove NaNs and clamp
    return (x <= pq_c1) ? 0.0f : sign(pq) * pow(abs((x - pq_c1) / (pq_c2 - pq_c3 * x)), 1.0f / pq_m1) * pq_C; // absolute of absolute is positive but fxc won't shut up without it.
}

float3 ZCAM_PQ_to_Linear(const in float3 pq)
{
    float3 x = pow(abs(pq), 1.0f / 134.034375f);
    x = min(max(x, 0.0f), 1.0087f); // Remove NaNs and clamp
    return (x <= pq_c1) ? 0.0f : sign(pq) * pow(abs((x - pq_c1) / (pq_c2 - pq_c3 * x)), 1.0f / pq_m1) * pq_C; // Absolute of absolute is positive but fxc won't shut up without it.
}

float Linear_to_ZCAM_PQ(const in float lin)
{
    float plin = pow(abs(lin) / pq_C, pq_m1);
    return (lin == 0.0f) ? 0.0f : sign(lin) * pow((pq_c1 + pq_c2 * plin) / (1.0f + pq_c3 * plin), 134.034375f);
}

float3 Linear_to_ZCAM_PQ(const in float3 lin)
{
    float3 plin = pow(abs(lin) / pq_C, pq_m1);
    return (lin == 0.0f) ? 0.0f : sign(lin) * pow((pq_c1 + pq_c2 * plin) / (1.0f + pq_c3 * plin), 134.034375f);
}

// Convert Iz to luminance
// Note that the PQ function used for Iz differs from the ST2084 function by replacing m_2 with rho.
// It also includes a luminance shift caused by the 2nd row-sum of the XYZ to LMS matrix not adding up to 1.0
float IzToLuminance(const in float Iz)
{
    float luminance = Iz <= 0.0f ? 0.0f : ZCAM_PQ_to_Linear(Iz) * 1.0285528f;
    return luminance;
}

// Convert luminance to Iz
// Note that the PQ fuction used for Iz differs from the ST2084 function by replacing m_2 with rho
// It also includes a luminance shift caused by the 2nd row-sum of the XYZ to LMS matrix not adding up to 1.0
float LuminanceToIz(const in float luminance)
{
    float Iz = luminance <= 0.0f ? 0.0f : Linear_to_ZCAM_PQ(luminance / 1.0285528f);
    return Iz;
}

// convert XYZ tristimulus values to the ZCAM intermediate Izazbz colorspace
float3 XYZ_to_Izazbz(const in float3 XYZD65)
{
    // To LMS space
    float3 XYZpD65 = XYZD65;
    XYZpD65.x = 1.15f * XYZD65.x - 0.15f * XYZD65.z;
    XYZpD65.y = 0.7f * XYZD65.y + 0.3f * XYZD65.x;        // Official coefficient for ZCAM here would be 0.66 instead of 0.7.
    float3 LMS = mul(XYZ_2_LMS_ZCAM_MAT, XYZpD65);
    float3 LMSp;
    LMSp = Linear_to_ZCAM_PQ(LMS);
    float3 Izazbz = mul(LMS_ZCAM_2_Izazbz_MAT, LMSp);
    Izazbz.x = max(Izazbz.x, 0.f);
    return Izazbz;
}

// convert the ZCAM intermediate Izazbz colorspace to XYZ tristimulus values
float3 Izazbz_to_XYZ(const in float3 Izazbz)
{
    float3 LMSp = mul(Izazbz_2_LMS_ZCAM_MAT, Izazbz);
    float3 LMS;
    LMS = ZCAM_PQ_to_Linear(LMSp);
    float3 XYZpD65 = mul(LMS_ZCAM_2_XYZ_MAT, LMS);
    float3 XYZD65 = XYZpD65;
    XYZD65.x = (XYZpD65.x + 0.15f * XYZpD65.z) / 1.15f;
    XYZD65.y = (XYZpD65.y - 0.3f * XYZD65.x) / 0.7f;
    return XYZD65;
}

float Radians_to_degrees(const in float radians)
{
    return radians * 180.0f / PI;
}

float Degrees_to_radians(const in float degrees)
{
    return degrees / 180.0f * PI;
}

// convert the ZCAM intermediate Izazbz colorspace to the ZCAM J (lightness), M (colorfulness) and h (hue) correlates
// needs the Iz values of the reference white and the viewing conditions parameters
float3 Izazbz_to_JMh(const in float3 Izazbz, const in float refWhiteIz, const in float F_b, const in float F_L, const in float F_s)
{
    float3 JMh;
    JMh.z = all(Izazbz.yz == 0.0f) ? 0.0f : fmod(Radians_to_degrees(atan2(Izazbz.z, Izazbz.y)) + 360.0f, 360.0f);
    float ez = 1.015f + cos(Degrees_to_radians(89.038f + JMh.z));
    float zp = 1.6f * F_s / pow(F_b, 0.12f);
    JMh.x = 100.0f * spow(Izazbz.x, zp) / pow(refWhiteIz, zp);
    JMh.y = 100.0f * pow(Izazbz.y * Izazbz.y + Izazbz.z * Izazbz.z, 0.37f) * ((pow(abs(ez), 0.068f) * pow(F_L, 0.2f)) / (pow(F_b, 0.1f) * pow(refWhiteIz, 0.78f)));

    return JMh;
}

// convert the ZCAM J (lightness), M (colorfulness) and h (hue) correlates to the ZCAM intermediate Izazbz colorspace
// needs the Iz values of the reference white and the viewing conditions parameters
float3 JMh_to_Izazbz(const in float3 JMh, const in float refWhiteIz, const in float F_b, const in float F_L, const in float F_s)
{
    float Qzw = pow(refWhiteIz, (1.6f * F_s) / pow(F_b, 0.12f));
    float Izp = pow(F_b, 0.12f) / (1.6f * F_s);
    float ez = 1.015f + cos(Degrees_to_radians(89.038f + JMh.z));
    float hzr = Degrees_to_radians(JMh.z);
    float Czp = spow((JMh.y * pow(refWhiteIz, 0.78f) * pow(F_b, 0.1f)) / (100.0f * pow(abs(ez), 0.068f) * pow(F_L, 0.2f)), 50.0f / 37.0f);

    return float3(spow((JMh.x * Qzw) / 100.0f, Izp), Czp * cos(hzr), Czp * sin(hzr));
}

// convert XYZ tristimulus values to the ZCAM J (lightness), M (colorfulness) and h (hue) correlates
// needs XYZ tristimulus values for the reference white and a D65 white as well as the viewing conditions as parameters
float3 XYZ_to_ZCAM_JMh(const in float3 XYZ, const in float refLum, const in float F_b, const in float F_L, const in float F_s)
{
    float3 refWhiteIzazbz = XYZ_to_Izazbz(D65_ReferenceWhite * refLum / D65_ReferenceWhite.y);
    return Izazbz_to_JMh(XYZ_to_Izazbz(XYZ), refWhiteIzazbz.x, F_b, F_L, F_s);
}

// convert the ZCAM J (lightness), M (colorfulness) and h (hue) correlates to XYZ tristimulus values
// needs XYZ tristimulus values for the reference white and a D65 white as well as the viewing conditions as parameters
float3 ZCAM_JMh_to_XYZ(const in float3 JMh, const in float refLum, const in float F_b, const in float F_L, const in float F_s)
{
    float3 refWhiteIzazbz = XYZ_to_Izazbz(D65_ReferenceWhite * refLum / D65_ReferenceWhite.y);
    return Izazbz_to_XYZ(JMh_to_Izazbz(JMh, refWhiteIzazbz.x, F_b, F_L, F_s));
}

float CompressPowerP(const in float v, const in float threshold, const in float limit, const in float power)
{
    float s = (limit - threshold) / pow(pow((1.0f - threshold) / (limit - threshold), -power) - 1.0f, 1.0f / power);

    float vCompressed;
    vCompressed = (v < threshold || limit < 1.0001f) ? v : threshold + s * ((v - threshold) / s) / (pow(1.0f + pow(abs(v - threshold) / s, power), 1.0f / power)); // Shut up fxc with abs(v - threshold). Can't be negative.

    return vCompressed;
}

float3 CompressPowerP(const in float3 v, const in float3 threshold, const in float3 limit, const in float power)
{
    float3 s = (limit - threshold) / pow(pow((1.0f - threshold) / (limit - threshold), -power) - 1.0f, 1.0f / power);

    float3 vCompressed;
    vCompressed = (v < threshold || limit < 1.0001f) ? v : threshold + s * ((v - threshold) / s) / (pow(1.0f + pow((abs(v - threshold)) / s, power), 1.0f / power)); // abs(v - threshold) to shut up fxc. Will never be negative.

    return vCompressed;
}

float3 AcesCompress(const in float3 linAP0)
{
    // Achromatic axis
    float ach = max(linAP0.x, max(linAP0.y, linAP0.z));
    float3 dist = (ach - linAP0) / abs(ach);

    // Compressed distance
    float3 compressedDist = CompressPowerP(dist, float3(0.815f, 0.803f, 0.990f), float3(1.147f, 1.264f, 1.312f), 1.2f);

    return ach - compressedDist * abs(ach);
}

float3 Sweeteners(const in float3 linAP1)
{
    float3 factors;
    factors = (TS_PARAMS_OUTPUT_DEVICE == 0) ? float3(-0.1f, 0.15f, 0.0f) : float3(0.0f, 0.15f, -0.1f);

    float3 linAP0 = mul(AP1_2_AP0_MAT, AcesCompress(linAP1));

    // Powers
    float globalPower = factors.x + 1.0f;
    float bluePower = (factors.y + 1.0f) * (factors.x + 1.0f);
    float cyanPower = (factors.z + 1.0f); // We use 1.0 instead of (factors.x + 1.0) for the global power in the high range.

    float maxNorm = max(linAP0.x, max(linAP0.y, linAP0.z));
    float3 r = (maxNorm <= 0.0f) ? float3(0.0f, 0.0f, 0.0f) : linAP0 / maxNorm;

    // max(r,g,b) norm and hue
    float h = rgb_to_hsv(r).x * 6.0f;

    // Dark adjustment factor
    float blueness = saturate(h - 3.0f) - saturate(h - 4.0f); // This gives a function shaped like this ___/\___ with peak at 4 == 240/60 (peak blue) and falloff 1 on both sides.
    bool hueIsEven = (((uint)(h + 1e-06f) & 1) == 0);
    float prim = (hueIsEven ? -1.0f : 1.0f) * frac(h) + (hueIsEven ? 1.0f : 0.0f); // Sawtooth pattern with a maximum of 1 when trunc(h) is even and a minimum of 0 when trunc(h) is odd.

    float3 lowAP0 = (r <= 0.0f) ? r : pow(abs(r), globalPower) * (prim - blueness) + pow(abs(r), bluePower) * blueness + r * (1.0f - prim); // Shut up fxc with abs(r)

    // The following lerp is an inherited bug that was accepted (or maybe not noticed) by artists. We can probably fix it over the summer
	// since it currently only applies to SDR.
    lowAP0 = (r <= 0.0f) ? r : lerp(pow(abs(lowAP0), globalPower), lowAP0, prim); // Shut up fxc with abs(lowAP0). Can't be negative or 0.
    lowAP0 *= maxNorm;

    // lerp lows
	float f = maxNorm <= 0.0f ? 0.0f : saturate(maxNorm / (maxNorm * maxNorm + maxNorm + 0.005f));
    f *= f;

    linAP0 = lerp(linAP0, lowAP0, f);

    // Update norm and hue
    maxNorm = max(linAP0.x, max(linAP0.y, linAP0.z));

    r = (maxNorm <= 0.0f) ? float3(0.0f, 0.0f, 0.0f) : linAP0 / maxNorm;
    h = rgb_to_hsv(r).x * 6.0f;

    // Bright adjustment factor
    float cyanness = saturate(h - 2.0f) - saturate(h - 3.0f); // Same logic as for blueness but peak at 3 == 180/60 (peak cyan).
    float3 highAP0 = r <= 0.0f ? r : lerp(r, pow(abs(r), cyanPower), cyanness); // Shut up fxc with abs(r)
    highAP0 *= maxNorm;

    // lerp highs
	f = maxNorm + 1.0f;
    f = maxNorm <= 0.0f ? 0.0f : saturate(1.0f - rcp(f*f));

    linAP0 = lerp(linAP0, highAP0, f);
    return linAP0;
}

float HighlightDesatFactor(const in float Iz, const in float IzTS, const in float refLum, const in TsParams tsParams)
{
    float linearLum = IzToLuminance(Iz) / refLum;
    if (linearLum < 0.18f)
        return 1.0f;

    const float real_gmThresh = tsParams.OutputDevice == 1 ? gmThresh - 0.05f : gmThresh; // HDR needs more desaturation
    const float hlDesat = tsParams.OutputDevice == 0 ? 3.5f : 4.0f; // Highlights need more desaturation in HDR

    float IzMid = LuminanceToIz(0.18f * refLum);
    float IzMidTS = LuminanceToIz(tsParams.MidLum);

    float IzAligned = Iz + IzMidTS - IzMid;

    float desatFactor = saturate(CompressPowerP((log10(max(HALF_MIN, IzAligned)) - log10(max(HALF_MIN, IzTS))) * hlDesat,
        real_gmThresh, HALF_POS_INF, gmPower));
    desatFactor = desatFactor > 0.8f ? (-1.0f / ((desatFactor - 0.8f) / (1.0f - 0.8f) + 1.0f) + 1.0f) * (1.0f - 0.8f) + 0.8f : desatFactor;

    return 1.0f - desatFactor;
}

float OldHighlightDesatFactor(const in float Iz, const in float IzTS, const in TsParams tsParams)
{
    const float HACK_hlDesat_SDR_reds_v9 = 2.75f;

    // Bringing this back is 100% a hack.
    float lum = IzToLuminance(IzTS);
    float desatFactor = saturate((lum - tsParams.MidLum) / tsParams.MidLum) * (Iz - IzTS);
    desatFactor *= HACK_hlDesat_SDR_reds_v9;
    desatFactor = desatFactor > 0.8f ? ( -1.0f / (( desatFactor - 0.8f ) / ( 1.0f - 0.8f ) + 1.0f ) + 1.0f ) * ( 1.0f - 0.8f ) + 0.8f : desatFactor;
    return 1.0f - desatFactor;
}

float ssts(float x, in TsParams C)
{
    // Textbook monomial to basis-function conversion matrix.
    const float3x3 M1 =
    {
        {  0.5f, -1.0f, 0.5f },
        { -1.0f,  1.0f, 0.5f },
        {  0.5f,  0.0f, 0.0f }
    };    
    
    const int N_KNOTS_LOW = 4;
    const int N_KNOTS_HIGH = 4;

    // Check for negatives or zero before taking the log. If negative or zero,
    // set to 1e-10f.
    float logx = log10( max(x, exp2(-14.0f) )); 
    float logy;

    if ( logx <= C.Min.x ) 
    { 
        logy = logx * C.SlopeLow + ( C.Min.y - C.SlopeLow * C.Min.x );
    } 
    else if (( logx > C.Min.x ) && ( logx < C.Mid.x )) 
    {
        float knot_coord = (N_KNOTS_LOW-1) * (logx-C.Min.x)/(C.Mid.x-C.Min.x);
        int j = (int)knot_coord;
        float t = knot_coord - j;

        float3 cf = { C.CoefsLow[ j], C.CoefsLow[ j + 1], C.CoefsLow[ j + 2]};

        float3 monomials = { t * t, t, 1.0f };
        logy = dot( monomials, mul( cf, M1));
    } 
    else if (( logx >= C.Mid.x ) && ( logx < C.Max.x )) 
    {
        float knot_coord = (N_KNOTS_HIGH-1) * (logx-C.Mid.x)/(C.Max.x-C.Mid.x);
        int j = (int)knot_coord;
        float t = knot_coord - j;

        float3 cf = { C.CoefsHigh[ j], C.CoefsHigh[ j + 1], C.CoefsHigh[ j + 2]}; 

        float3 monomials = { t * t, t, 1.0f };
        logy = dot( monomials, mul( cf, M1));
    } 
    else 
    { //if ( logIn >= log10(C.Max.x) ) { 
        logy = logx * C.SlopeHigh + ( C.Max.y - C.SlopeHigh * C.Max.x );
    }

    return pow(10.f, logy);
}

float3 ForwardTonescale(const in float3 inputIzazbz, const in float refLum, const in float F_b,
    const in float F_L, const in float F_s, const in float blendFactor, const in TsParams tsParams)
{
    float3 refWhiteIzazbz = XYZ_to_Izazbz(D65_ReferenceWhite * refLum / D65_ReferenceWhite.y);

    float linearLum = IzToLuminance(inputIzazbz.x) / refLum;
    float luminanceTS = ssts(linearLum, tsParams);
    float IzTS = LuminanceToIz(luminanceTS);

    float3 outputIzazbz = inputIzazbz;
    outputIzazbz.x = IzTS;

    // convert the result to JMh
    float3 outputJMh = Izazbz_to_JMh(outputIzazbz, refWhiteIzazbz.x, F_b, F_L, F_s);

    float factM = HighlightDesatFactor(inputIzazbz.x, IzTS, refLum, tsParams);
	
	// Less highlight desaturation for targeted hues in SDR. The parameters were optimized to prevent the fire from
	// turning pink while not harming the skin tones too much.
    factM = lerp(factM, OldHighlightDesatFactor(inputIzazbz.x, IzTS, tsParams), blendFactor);

    outputJMh.y = outputJMh.y * factM;

    return outputJMh;
}

// check if the 3D point 'v' is inside a cube with the dimensions cubeSize x cubeSize x cubeSize
// the 'smoothing' parameter rounds off the edges and corners of the cube with the exception of the 0,0,0 and cubeSize x cubeSize x cubeSize corners
// a smoothing value of 0.0 applies no smoothing and 1.0 the maximum amount (smoothing values > 1.0 result in undefined behavior )
bool IsInsideCube(const in float3 v, const in float cubeSize, const in float smoothing)
{
    float3 normv = v / cubeSize;

    float minv = min(normv.x, min(normv.y, normv.z));
    float maxv = max(normv.x, max(normv.y, normv.z));

    [branch]
    if (smoothing <= 0.0f)
    {
        // when not smoothing we can use a much simpler test
        if (minv < 0.0f || maxv > 1.0f)
        {
            return false;
        }

        return true;
    }

    float3 clamped = normv;
    float radius = clamp(smoothing, 0.0f, 0.999f) / 2.0f;

    radius = clamp(radius * maxv * (1.0f - minv), 0.0f, radius);
    clamped = clamp(normv, radius, 1.0f - radius);

    float3 diff = normv - clamped;
    float lenSqr = dot(diff, diff);
    if (lenSqr > radius * radius)
    {
        return false;
    }

    return true;
}

float2 EstimateGamutCusp(const in float3 inputJMh, const in float refLum, const in float F_b, const in float F_L,
    const in float F_s, const in TsParams tsParams)
{
    float3 limitRGB = mul(GetLimitingPrimaries_XYZ_2_RGB_Matrix(tsParams.OutputDevice), ZCAM_JMh_to_XYZ(inputJMh, refLum, F_b, F_L, F_s)) / refLum;

    // Clip to boundary
    if (any(limitRGB < 0.0f))
    {
        float toSub = min(min(limitRGB.x, limitRGB.y), limitRGB.z);
        limitRGB -= toSub;
    }

    if (any(limitRGB > 1.0f))
    {
        float divider = max(max(limitRGB.x, limitRGB.y), limitRGB.z);
        limitRGB /= divider;
    }

    float h = rgb_to_hsv(limitRGB).x;
    const float3x3 LimitingPrimaries_RGB_2_XYZ_Matrix = GetLimitingPrimaries_RGB_2_XYZ_Matrix(tsParams.OutputDevice);

    float3 rgbCuspTest = hsv_to_rgb(float3(h, 1.0f, 1.0f));
    float3 JMhCuspTest = XYZ_to_ZCAM_JMh(mul(LimitingPrimaries_RGB_2_XYZ_Matrix, rgbCuspTest * tsParams.MaxLum), refLum, F_b, F_L, F_s);

    [branch]
    if (JMhCuspTest.z == inputJMh.z)
        return JMhCuspTest.xy;

    float direction = (JMhCuspTest.z < inputJMh.z) ? 1.0f : -1.0f;

    [loop]
    for (int i = 0; i < 360; i++)
    {
        h += direction / 360.0f;

        float3 newJMhCuspTest = XYZ_to_ZCAM_JMh(mul(LimitingPrimaries_RGB_2_XYZ_Matrix, hsv_to_rgb(float3(h, 1.0f, 1.0f)) * tsParams.MaxLum),
            refLum, F_b, F_L, F_s);
        float newDirection = (newJMhCuspTest.z == inputJMh.z) ? 0.0f : (newJMhCuspTest.z < inputJMh.z) ? 1.0f : -1.0f;

        [branch]
        if (newDirection != direction || h < 0.0f || h > 1.0f)
        {
            float3 lo = (JMhCuspTest.z <= newJMhCuspTest.z) ? JMhCuspTest : newJMhCuspTest;
            float3 hi = (JMhCuspTest.z <= newJMhCuspTest.z) ? newJMhCuspTest : JMhCuspTest;

            float t = saturate((inputJMh.z - lo.z) / (hi.z - lo.z));
            return lerp(lo.xy, hi.xy, t);
        }

        JMhCuspTest = newJMhCuspTest;
    }

    return JMhCuspTest.xy;
}

// find the JM coordinates of the smoothed boundary of the limiting gamut in ZCAM at the hue slice 'h'
// by searching along the line defined by 'JMSource' and 'JMFocus'
// the function will search outwards from where the line intersects the achromatic axis with a staring incement of 'startStepSize'
// once the boundary has been crossed it will search in the opposite direction with half the step size
// and will repeat this as as many times as is set by the 'precision' parameter
float2 FindBoundary(const in float2 JMSource, const in float2 JMFocus, const in float h, const in float refLum,
    const in float F_b, const in float F_L, const in float F_s, const in int outputDevice, const in float startStepSize,
    const in float boundarySize, const in float smoothing, const in float limitJmax, const in float limitMmax)
{
    float2 achromaticIntercept = float2(JMFocus.x - (((JMSource.x - JMFocus.x) / (JMSource.y - JMFocus.y)) * JMFocus.y), 0.0f);

    [branch]
    if (achromaticIntercept.x <= 0.0f || achromaticIntercept.x >= limitJmax || all(achromaticIntercept == JMFocus))
    {
        return achromaticIntercept;
    }

    float stepSize = startStepSize;
    float2 unitVector = normalize(achromaticIntercept - JMFocus);
    float2 JMtest = achromaticIntercept;
    bool searchOutwards = true;

    const float3x3 LimitingPrimaries_XYZ_2_RGB_Matrix = GetLimitingPrimaries_XYZ_2_RGB_Matrix(outputDevice);

    [loop]
    for (int i = 0; i < 10; ++i)
    {
        // Changed the original while(1) to a loop with finite iterations. It should never take more than 128 iterations
        // to converge and, if it does, then we can accept that it might never have converged in any case.
        [loop]
        for (int j = 0; j < 128; ++j)
        {
            JMtest = JMtest + unitVector * stepSize;
            bool inside = IsInsideCube(mul(LimitingPrimaries_XYZ_2_RGB_Matrix,
                ZCAM_JMh_to_XYZ(float3(JMtest.x, JMtest.y, h), refLum, F_b, F_L, F_s) / refLum), boundarySize, smoothing);

            [branch]
            if (searchOutwards && (JMtest.x < 0.0f || JMtest.x > limitJmax || JMtest.y > limitMmax || !inside))
            {
                searchOutwards = false;
                stepSize = -abs(stepSize) / 2.0f;
                break;
            }
            else if (!searchOutwards && (JMtest.y < 0.0f || inside))
            {
                searchOutwards = true;
                stepSize = abs(stepSize) / 2.0f;
                break;
            }
        }
    }

    float2 JMboundary = float2(clamp(JMtest.x, 0.0f, limitJmax), clamp(JMtest.y, 0.0f, limitMmax));

    return JMboundary;
}

float3 CompressGamut(const in float3 inputJMh, const in float refLum, const in float F_b,
    const in float F_L, const in float F_s, const in TsParams tsParams)
{
    const float real_gmThresh = tsParams.OutputDevice == 1 ? gmThresh - 0.05f : gmThresh; // HDR needs more desaturation

    const float3x3 LimitingPrimaries_RGB_2_XYZ_Matrix = GetLimitingPrimaries_RGB_2_XYZ_Matrix(tsParams.OutputDevice);

    // limitJmax (assumed to match limitRGB white)
    float limitJmax = XYZ_to_ZCAM_JMh(mul(LimitingPrimaries_RGB_2_XYZ_Matrix, float3(1.0f, 1.0f, 1.0f) * tsParams.MaxLum),
            refLum, F_b, F_L, F_s).x;

    // limitMmax (assumed to coincide with one of the RGBCMY corners of the limitRGB cube)
    float limitMmax = 0.0f;
    limitMmax = max(limitMmax, XYZ_to_ZCAM_JMh(mul(LimitingPrimaries_RGB_2_XYZ_Matrix, float3(1.0f, 0.0f, 0.0f) * tsParams.MaxLum),
        refLum, F_b, F_L, F_s).y);
    limitMmax = max(limitMmax, XYZ_to_ZCAM_JMh(mul(LimitingPrimaries_RGB_2_XYZ_Matrix, float3(1.0f, 1.0f, 0.0f) * tsParams.MaxLum),
        refLum, F_b, F_L, F_s).y);
    limitMmax = max(limitMmax, XYZ_to_ZCAM_JMh(mul(LimitingPrimaries_RGB_2_XYZ_Matrix, float3(0.0f, 1.0f, 0.0f) * tsParams.MaxLum),
        refLum, F_b, F_L, F_s).y);
    limitMmax = max(limitMmax, XYZ_to_ZCAM_JMh(mul(LimitingPrimaries_RGB_2_XYZ_Matrix, float3(0.0f, 1.0f, 1.0f) * tsParams.MaxLum),
        refLum, F_b, F_L, F_s).y);
    limitMmax = max(limitMmax, XYZ_to_ZCAM_JMh(mul(LimitingPrimaries_RGB_2_XYZ_Matrix, float3(0.0f, 0.0f, 1.0f) * tsParams.MaxLum),
        refLum, F_b, F_L, F_s).y);
    limitMmax = max(limitMmax, XYZ_to_ZCAM_JMh(mul(LimitingPrimaries_RGB_2_XYZ_Matrix, float3(1.0f, 0.0f, 1.0f) * tsParams.MaxLum),
        refLum, F_b, F_L, F_s).y);

    // estimate the gamut cusp in JMh to avoid the need for building a table then
    // doing a linear search into that table
    float2 JMinput = inputJMh.xy;
    float2 JMcusp = EstimateGamutCusp(inputJMh, refLum, F_b, F_L, F_s, tsParams);

    float sstsMidJ = XYZ_to_ZCAM_JMh(D65_ReferenceWhite * tsParams.MidLum, refLum, F_b, F_L, F_s).x;
    float focusJ = lerp(JMcusp.x, sstsMidJ, gmCuspMidBlend);

    float focusDistanceGain = 1.0f;

    if (JMinput.x > focusJ)
    {
        focusDistanceGain = (limitJmax - focusJ) / max(0.0001f, (limitJmax - min(limitJmax, JMinput.x)));
    }
    else
    {
        focusDistanceGain = focusJ / max(0.0001f, JMinput.x);
    }

    float2 JMfocus = float2(focusJ, -JMcusp.y * gmFocusDistance * focusDistanceGain);
    float2 vecToFocus = JMfocus - JMinput;
    float2 achromaticIntercept = float2(JMfocus.x - (((JMinput.x - JMfocus.x) / (JMinput.y - JMfocus.y)) * JMfocus.y), 0.0f);

    // to reduce the number of expensive boundary finding iterations needed
    // we taking an educated guess at a good starting step size
    // based on how far the sample is either above or below the gamut cusp
    float cuspToTipRatio = 0.0f;
    if (JMinput.x > JMcusp.x)
    {
        cuspToTipRatio = saturate((JMinput.x - JMcusp.x) / max((limitJmax - JMcusp.x), 1e-06f));
    }
    else
    {
        cuspToTipRatio = saturate((JMcusp.x - JMinput.x) / max(JMcusp.x, 1e-06f));
    }

    float startStepSize = lerp(JMcusp.y / 3.0f, 0.1f, cuspToTipRatio);
    float2 JMboundary = FindBoundary(JMinput, JMfocus, inputJMh.z, refLum, F_b, F_L, F_s, tsParams.OutputDevice,
        startStepSize, tsParams.MaxLum / refLum, gmSmoothCusps, limitJmax, limitMmax);
    float normFact = 1.0f / max(0.0001f, length(JMboundary - achromaticIntercept));
    float v = length(JMinput - achromaticIntercept) * normFact;
    float vCompressed = CompressPowerP(v, real_gmThresh, gmLimit, gmPower);
    float2 JMcompressed = all(JMinput == achromaticIntercept) ? achromaticIntercept :
        achromaticIntercept + normalize(JMinput - achromaticIntercept) * vCompressed / normFact;
    return float3(JMcompressed.x, JMcompressed.y, inputJMh.z);
}

float3 OldCompressGamut(const in float3 inputJMh, const in float refLum, const in float F_b,
    const in float F_L, const in float F_s, const in int outputDevice)
{
    const float HACK_gmThresh_SDR_reds_v9 = 0.68f;
    const float HACK_gmLimit_SDR_reds_v9 = 1.35f;
    const float HACK_gmPower_SDR_reds_v9 = 1.075f;

    float3 outputJMh = inputJMh;
    float bound = -1.0f;

    const float3x3 LimitingPrimaries_XYZ_2_RGB_Matrix = GetLimitingPrimaries_XYZ_2_RGB_Matrix(outputDevice);

    [loop]
    for( int i = 0; i < 128; ++i)
    {
        outputJMh.y = (float)i + 1.0f;
        float3 RGB_test = mul(LimitingPrimaries_XYZ_2_RGB_Matrix, ZCAM_JMh_to_XYZ(outputJMh, refLum, F_b, F_L, F_s));

        [branch]
        if (any(RGB_test < 0.0f))
        {
            bound = outputJMh.y;
            break;
        }
    }

    bound = max(bound, 0.0f);
    bound = bound > 2.0f ? bound : 1.0f + bound * bound / 4.0f;

    outputJMh.y = (inputJMh.y >= 1e-06f) ? max(bound * CompressPowerP(inputJMh.y / bound, HACK_gmThresh_SDR_reds_v9,
        HACK_gmLimit_SDR_reds_v9, HACK_gmPower_SDR_reds_v9), 1e-06f) : inputJMh.y;
    return outputJMh;
}

void FillTsParams(out TsParams tsParams)
{
    tsParams.Min = float2(TS_PARAMS_MIN_X, TS_PARAMS_MIN_Y);
    tsParams.Mid = float2(TS_PARAMS_MID_X, TS_PARAMS_MID_Y);
    tsParams.Max = float2(TS_PARAMS_MAX_X, TS_PARAMS_MAX_Y);
    tsParams.SlopeLow = TS_PARAMS_MIN_SLOPE;
    tsParams.SlopeHigh = TS_PARAMS_MAX_SLOPE;

    tsParams.CoefsLow[0] = TS_PARAMS_COEFS_LOW_0;
    tsParams.CoefsLow[1] = TS_PARAMS_COEFS_LOW_1;
    tsParams.CoefsLow[2] = TS_PARAMS_COEFS_LOW_2;
    tsParams.CoefsLow[3] = TS_PARAMS_COEFS_LOW_3;
    tsParams.CoefsLow[4] = TS_PARAMS_COEFS_LOW_4;
    tsParams.CoefsLow[5] = TS_PARAMS_COEFS_LOW_5;
    // Should be only 6 coefficients, but there's a driver issue with Vulkan on AMD
    tsParams.CoefsLow[6] = 0;
    tsParams.CoefsLow[7] = 0;

    tsParams.CoefsHigh[0] = TS_PARAMS_COEFS_HIGH_0;
    tsParams.CoefsHigh[1] = TS_PARAMS_COEFS_HIGH_1;
    tsParams.CoefsHigh[2] = TS_PARAMS_COEFS_HIGH_2;
    tsParams.CoefsHigh[3] = TS_PARAMS_COEFS_HIGH_3;
    tsParams.CoefsHigh[4] = TS_PARAMS_COEFS_HIGH_4;
    tsParams.CoefsHigh[5] = TS_PARAMS_COEFS_HIGH_5;
    // Should be only 6 coefficients, but there's a driver issue with Vulkan on AMD
    tsParams.CoefsHigh[6] = 0;
    tsParams.CoefsHigh[7] = 0;

    tsParams.MinLum = TS_PARAMS_MIN_LUM;
    tsParams.MidLum = TS_PARAMS_MID_LUM;
    tsParams.MaxLum = TS_PARAMS_MAX_LUM;
    tsParams.HDRPaperWhite = TS_PARAMS_HDR_PAPER_WHITE;
    tsParams.HDRGamma = TS_PARAMS_HDR_GAMMA;
    tsParams.OutputDevice = TS_PARAMS_OUTPUT_DEVICE;
}

float GetBlendFactor(float3 colorAP0)
{
    // NB: These is no rationale to these values. It was all curve fitted in Desmos then visually matched.
	const float pivotHueRedToGreen0 = 30.5f;
	const float pivotHueRedToGreen1 = 30.84f;
	const float pivotHuePurpleToRed0 = 325.0f;
	const float pivotHuePurpleToRed1 = 352.0f;
	const float widthRedToGreen = 92.6f;
	const float widthPurpleToRed = 50.0f;

    float hueAP0 = rgb_to_hsv(max(colorAP0, 0.0f)).x * 360.0f;

	// 1.25x boost to overestimate red->green blend factor since the curve was fitted on low intensity colors
	// and the resulting blend factor is too low for high intensity colors. It's less harmful that way.
	float f0 = 1.25f * smoothstep(widthRedToGreen - 1.5f * hueAP0, 0.0f, widthRedToGreen);
	
	// No 1.25x boost here or it won't connect.
	float p0 = smoothstep(widthRedToGreen - 1.5f * pivotHueRedToGreen1, 0.0f, widthRedToGreen) * (pivotHueRedToGreen1 + widthRedToGreen);

	float f1 = 1.25f * smoothstep(p0 - hueAP0, -pivotHueRedToGreen1, widthRedToGreen);
	float redToGreen = saturate(lerp(f0, f1, saturate((hueAP0 - pivotHueRedToGreen0) * 2.0f)));

	float g0 = smoothstep(hueAP0, pivotHuePurpleToRed0, pivotHuePurpleToRed0 + widthPurpleToRed);
    float g1 = saturate((hueAP0 - pivotHuePurpleToRed1) / widthPurpleToRed) + g0;
	
	// FIXME: redToGreen is close enough to original node in Resolve but purpleToRed hits 1.0 too early. Still better than the zcam_4.dds LUT in Perforce.
    float purpleToRed = hueAP0 < pivotHuePurpleToRed1 ? g0 : g1;

    // No hacking when chroma is too low as rgb_to_hsv returns red hue with very low saturation in this case.
    float blendFactor = saturate(redToGreen + purpleToRed);
    blendFactor = (max(colorAP0.x, max(colorAP0.y, colorAP0.z)) - min(colorAP0.x, min(colorAP0.y, colorAP0.z)) < 0.001f) ? 0.0f : blendFactor;

    return blendFactor;
}

float3 AdjustHDRContrast(float3 x, float gamma)
{
    // Reference: Microsoft XDK HDRCalibration sample
    
    // Adjust HDR contrast. For SDR in the range [0..1], we can safely apply a simple pow() function, but for HDR, values higher than 1.0f will also be effected.
    // This function interpolates between the power function and the linear HDR value to ensure that when reducing contrast, the maximum brightness in the scene
    // is not getting dimmer.
	float3 xGamma = (x <= 0.0f) ? 0.0f : pow(abs(x), gamma);
	
    if (gamma >= 1.0f)
    {
        return xGamma;
    }
    else
    {
        float3 t = saturate((2.0f - x) * (2.0f - x) * (2.0f - x));
        float3 y = (x * (1.0f - t)) + (xGamma * t);
        return y;
    }
}

[numthreads(16, 16, 2)]
void main( uint3 DispatchThreadID : SV_DispatchThreadID )
{
    TsParams tsParams;
    FillTsParams(tsParams);

    // We assume dim viewing conditions (sRGB standard default), i.e. an ambient light level of 64 lux. Unlike for earlier implementations,
    // we will stick to that assumption for HDR too as it makes matching brightness of SDR and HDR easier. Testing of a bright
    // high-quality 1000 nits HDR screen in dark viewing conditions (ambient light levels of 5 lux or less) has proven eye searing in any case.
    const float refLum = (tsParams.OutputDevice == 0) ? 125.0f : 200.0f;
    const float F = 0.9f;
    const float F_s = 0.59f;
    const float F_s_in = 0.525f;
    const float L_A = refLum * refLum / 1000.0f;
    const float adaptDegree_in = 1.0f;
    const float F_b = 0.316227764f;
    const float F_L = 0.171f * pow(L_A, 1.0f / 3.0f) * (1.0f - exp(-48.0f / 9.0f * L_A));

    // Convert thread ID to volume LUT coord
    float3 uvw = (float3)DispatchThreadID.xyz / float3(63.0f, 63.0f, 63.0f);

    // TODO: AcesCc as a shaper doesn't optimally fill the LUT for nits levels lower than 1000 and may not be
	//       the best shaper for 1000 nits either but it seems to fill it better than scaled PQ. There might be
	//       too much emphasis on blacks though and yellows don't seem evenly distributed. Investigate.
    float3 colorAP1 = AcesCc_2_Linear(uvw);
    float3 colorAP0 = Sweeteners(colorAP1);


    // In SDR, use alternate Y_MAX of 125.0 instead of 100.0 on region centered on red.
    // This will introduce clipping that is useful for fire.
    float blendFactor = (tsParams.OutputDevice == 1) ? 0.0f : GetBlendFactor(colorAP0);

    tsParams.CoefsHigh[0] = lerp(tsParams.CoefsHigh[0], HACK_SDR_AlternateHighCoeffs.x, blendFactor);
    tsParams.CoefsHigh[1] = lerp(tsParams.CoefsHigh[1], HACK_SDR_AlternateHighCoeffs.y, blendFactor);
    tsParams.CoefsHigh[2] = lerp(tsParams.CoefsHigh[2], HACK_SDR_AlternateHighCoeffs.z, blendFactor);
    tsParams.CoefsHigh[3] = lerp(tsParams.CoefsHigh[3], HACK_SDR_AlternateHighCoeffs.w, blendFactor);
    tsParams.CoefsHigh[4] = lerp(tsParams.CoefsHigh[4], HACK_SDR_AlternateHighCoeffs.w, blendFactor);	// Assumes a slope at MaxPt of 0 -> CoefsHigh[3..5] are all equal
    tsParams.CoefsHigh[5] = lerp(tsParams.CoefsHigh[5], HACK_SDR_AlternateHighCoeffs.w, blendFactor);
    tsParams.Min.x = lerp(tsParams.Min.x, HACK_SDR_AlternateTsPoints.x, blendFactor);
    tsParams.Mid.x = lerp(tsParams.Mid.x, HACK_SDR_AlternateTsPoints.y, blendFactor);
    tsParams.Max.x = lerp(tsParams.Max.x, HACK_SDR_AlternateTsPoints.z, blendFactor);
    tsParams.Max.y = lerp(tsParams.Max.y, HACK_SDR_AlternateTsPoints.w, blendFactor);
    tsParams.MaxLum = lerp(tsParams.MaxLum, pow(10.0f, HACK_SDR_AlternateTsPoints.w), blendFactor);

    float3 XYZ = mul(AP0_2_XYZ_MAT, colorAP0) * refLum;
    float3 Izazbz = XYZ_to_Izazbz(CAT_Zhai2018(XYZ, ACES_ReferenceWhite, D65_ReferenceWhite, adaptDegree_in, adaptDegree_in));

    float3 JMh = ForwardTonescale(Izazbz, refLum, F_b, F_L, F_s_in, blendFactor, tsParams);
    float3 JMhCompressed = CompressGamut(JMh, refLum, F_b, F_L, F_s, tsParams);

    [branch]
    if (tsParams.OutputDevice == 0)
    {
		// Use buggy gamut compression that ignores high limit for targeted hues in SDR. This reintroduces clipping for reds and
		// ensure that bright fire vfx turn orange instead of pink. It also prevents the brine bulbs in Avernus from going completely white (again).
		// The parameters were optimized to help fire vfx while not harming skin tones too much.
        JMhCompressed = lerp(JMhCompressed, OldCompressGamut(JMh, refLum, F_b, F_L, F_s, tsParams.OutputDevice), blendFactor);
    }

    // HACK: We will apply the lightness compression in linear XYZ space instead. This prevents the fake night atmosphere from
    //       shifting dark blue -> dark cyan. The latest version of the official implementation has re-optimized the ZCAM matrix in an
    //       attempt to achieve the same effect but it lifts dark blues too much and doesn't look as good on our data. We can revisit this
    //       for FW5.
    float scaleFactor = (JMh.x != 0.0f) ? JMhCompressed.x / JMh.x : 1.0f;
    JMhCompressed.x = JMh.x;

    XYZ = ZCAM_JMh_to_XYZ(JMhCompressed, refLum, F_b, F_L, F_s);
    XYZ *= scaleFactor;

    float3 output = max(0.0f, mul(GetOutputPrimaries_XYZ_2_RGB_Matrix(tsParams.OutputDevice), XYZ)); // max to get rid of NaNs and negatives
    if (tsParams.OutputDevice == 0)
    {
        output = Y_2_sRGB(output / 100.0f);
        //TODO: SDR brightness and gamma here
    }
    else
    {
        // HDR paper white and contrast
        output = AdjustHDRContrast(output / refLum, tsParams.HDRGamma) * refLum * tsParams.HDRPaperWhite;

        // 20% overshoot headroom allows monitor tonemapping to take over without too much adverse
        // effects on monitors that have low-quality dynamic dimming.
        output = Y_2_ST2084(clamp(output, 0.0f, tsParams.MaxLum * 1.2f));
    }

    LUT[DispatchThreadID] = float4(saturate(output), 1.0f);
}
