package com.example.ageclassifier;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.ageclassifier.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private static final String TAG = MainActivity.class.getSimpleName();
    public static Intent newIntent(Context context) { Log.d(TAG,"newIntent()");
        return new Intent(context.getApplicationContext(), MainActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
    }
    private static final int PERMISSION_STATE = 0;
    private static final int CAMERA_REQUEST = 1;
    private Button imgCamera;
    private ImageView imgResult;
    private Button btnPredict;
    private TextView txtPrediction;
    private Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) { Log.d(TAG,"onCreate()");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imgCamera = (Button) findViewById(R.id.captureButton);
        imgResult = (ImageView) findViewById(R.id.resultImage);
        txtPrediction = (TextView) findViewById(R.id.textViewPrediction);
        btnPredict = (Button) findViewById(R.id.scanButton);
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {

            case R.id.scanButton:
                predict();
                break;
            case R.id.captureButton:
                launchCamera();
                break;
            default:
                break;
        }
    }

    @Override
    protected void onResume() { Log.d(TAG,"onResume()");
        super.onResume();
        btnPredict.setOnClickListener(this::onClick);
        imgCamera.setOnClickListener(this::onClick);
        checkPermissions();
    }

    @Override
    protected void onPause() { Log.d(TAG,"onPause()");
        super.onPause();
        btnPredict.setOnClickListener(null);
        imgCamera.setOnClickListener(null);
    }

    private void launchCamera() { Log.d(TAG,"launchCamera()");
        startActivityForResult(new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE), CAMERA_REQUEST);
    }



    private void predict() { Log.d(TAG,"predict()");
        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        try { Log.d(TAG,"try");
            Model model = Model.newInstance(getApplicationContext());
            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);
            TensorImage tensorImage = new TensorImage(DataType.UINT8);
            tensorImage.load(bitmap);
            ByteBuffer byteBuffer = tensorImage.getBuffer();

            inputFeature0.loadBuffer(byteBuffer);
            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            // Releases model resources if no longer used.
            model.close();
            txtPrediction.setText(getMax(outputFeature0.getFloatArray()));//txtPrediction.setText(outputFeature0.getFloatArray()[0] + "\n" + outputFeature0.getFloatArray()[1] + "\n" + outputFeature0.getFloatArray()[2]);
            getMax(outputFeature0.getFloatArray());
            Log.d("Result",Arrays.toString(outputFeature0.getFloatArray()));
        } catch (IOException e) {
            Log.e(TAG,"IOException " + e.getMessage());
        }
    }

    private String getMax(float [] outputs) { Log.d(TAG,"getMax( " + Arrays.toString(outputs) + ")");
        if (outputs.length != 0 & outputs[0] > outputs[1] & outputs[0] > outputs[2] & outputs[0] > outputs[3] & outputs[0] > outputs[4] & outputs[0] > outputs[5] & outputs[0] > outputs[6] & outputs[0] > outputs[7]) {
            return "APPLE \n Apples have been linked to a lower risk of heart disease. \n An acre of apples will extract about 15 tons of carbon dioxide from the air each year, and produce 6 tons of oxygen too. \n Quercetin in apples may protect your brain from damage caused by oxidative stress. \n Calories: 104";
        } else if (outputs.length != 0 & outputs[1] > outputs[0] & outputs[1] > outputs[2] & outputs[1] > outputs[3] & outputs[1] > outputs[4] & outputs[1] > outputs[5] & outputs[1] > outputs[6] & outputs[1] > outputs[7]) {
            return "BANANA \n Bananas are a great food for anyone who cares about their carbon footprint. For just 80g of CO2e you get a whole lot of nutrition: 140 calories as well as stacks of vitamin C, vitamin B6, potassium and dietary fibre.  \n  Calories: 112";
        } else if (outputs.length != 0 & outputs[2] > outputs[0] & outputs[2] > outputs[1] & outputs[2] > outputs[3] & outputs[2] > outputs[4] & outputs[2] > outputs[5] & outputs[2] > outputs[6] & outputs[2] > outputs[7]) {
            return "LEMON \n The citric acid in lemons may reduce your risk of kidney stones. Lemons may help reduce the risk of many types of cancers, including breast cancer. This is thought to be due to plant compounds like hesperidin and d-limonene. \n Lemon juice is a natural weed killer. \n Calories: 29";
        }else if (outputs.length != 0 & outputs[3] > outputs[0] & outputs[3] > outputs[1]  & outputs[3] > outputs[2]& outputs[3] > outputs[4] & outputs[3] > outputs[5]  & outputs[3] > outputs[6] & outputs[3] > outputs[7]) {
            return "CARROT \n High blood cholesterol is a well-known risk factor for heart disease. Intake of carrots has been linked to lower cholesterol levels. \n Individuals with low vitamin A levels are more likely to experience night blindness, a condition that may diminish by eating carrots or other foods rich in vitamin A or carotenoids. \n Carrots have a low carbon footprint compared to other foods. What is the carbon footprint of carrots? It takes around 0.11 kg CO2e to produce 1 kilogram or 2.2 pounds of carrots, a car driving equivalent of 0.25 miles or 0.5 kilometers. \n Calories: 41";
        } else if (outputs.length != 0 & outputs[4] > outputs[0] & outputs[4] > outputs[1] & outputs[4] > outputs[2] & outputs[4] > outputs[3] & outputs[4] > outputs[5] & outputs[4] > outputs[6] & outputs[4] > outputs[7]) {
            return "POTATO \n Potatoes are a good source of antioxidants, including specific types, such as flavonoids, carotenoids and phenolic acids. One study compared the antioxidant activities of white and colored potatoes and found that colored potatoes were the most effective at neutralizing free radicals. \n Potatoes are a good source of fiber, which can help you lose weight by keeping you full longer. \n Calories: 168";
        } else if (outputs.length != 0 & outputs[5] > outputs[0] & outputs[5] > outputs[1] & outputs[5] > outputs[2] & outputs[5] > outputs[3] & outputs[5] > outputs[4] & outputs[5] > outputs[6] & outputs[5] > outputs[7]) {
            return "MINT \n Mint may also be effective at relieving other digestive problems such as upset stomach and indigestion. In addition to ingesting mint, there are claims that inhaling the aroma of essential oils from the plant could provide health benefits, including improved brain function. One study including 144 young adults demonstrated that smelling the aroma of peppermint oil for five minutes prior to testing produced significant improvements in memory. \n Mint Attracts Beneficial Insects (& Repels the Bad Ones) The smell of the mint plant will also repel houseflies, cabbage moths, ants, aphids, squash bugs, fleas, mosquitoes, and even mice. \n Calories: 6";
        }else if (outputs.length != 0 & outputs[6] > outputs[0] & outputs[6] > outputs[1]  & outputs[6] > outputs[2]& outputs[6] > outputs[3] & outputs[6] > outputs[4]  & outputs[6] > outputs[5] & outputs[6] > outputs[7]) {
            return "MANGOES \n Mango demonstrates some exciting potential when it comes to healthy weight control. Recent research suggests that mango and its phytochemicals may actually suppress fat cells and fat-related genes.  Another study showed that mango peel inhibits the formation of fatty tissues in a way similar to the antioxidant resveratrol. \n The micronutrients in mango may fight cancer, and research on breast cancer in particular is promising. In one animal studyTrusted Source, mango decreased tumor size and suppressed cancer growth factors. In another study, mango stopped the advancement of an early-stage breast cancer called ductal carcinoma. \n Calories: 107";
        }else if (outputs.length != 0 & outputs[7] > outputs[0] & outputs[7] > outputs[1]  & outputs[7] > outputs[2]& outputs[7] > outputs[3] & outputs[7] > outputs[4]  & outputs[7] > outputs[5] & outputs[7] > outputs[6]) {
            return "NOTHING \n Please try again the AI was not able to recognize the object/fruit/vegetable.";
        }else {
            return "";
        }
    }

    private void checkPermissions() {
        String[] manifestPermissions;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN) {
            manifestPermissions = new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            };
        } else {
            manifestPermissions = new String[] {
                    Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            };
        }
        for (String permission : manifestPermissions) {
            if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
                Log.d(TAG,"Permission Granted " + permission);
            }
            if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_DENIED) {
                Log.d(TAG,"Permission Denied " + permission);
                requestPermissions();
            }
        }
    }

    private void requestPermissions() { Log.d(TAG, "requestPermissions()");
        String[] manifestPermissions;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN) {
            manifestPermissions = new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            };
        } else {
            manifestPermissions = new String[] {
                    Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            };
        }

        ActivityCompat.requestPermissions(
                this,
                manifestPermissions,
                PERMISSION_STATE
        );
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        Log.d(TAG, "PermissionsResult requestCode " + requestCode);
        Log.d(TAG, "PermissionsResult permissions " + Arrays.toString(permissions));
        Log.d(TAG, "PermissionsResult grantResults " + Arrays.toString(grantResults));
        if (requestCode == PERMISSION_STATE) {
            for (int grantResult : grantResults) {
                switch (grantResult) {
                    case PackageManager.PERMISSION_GRANTED:
                        Log.d(TAG, "PermissionsResult grantResult Allowed " + grantResult);
                        break;
                    case PackageManager.PERMISSION_DENIED:
                        Log.d(TAG, "PermissionsResult grantResult Denied " + grantResult);
                        break;
                    default:
                        break;
                }
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Log.d(TAG, "onActivityResult requestCode " + requestCode + " resultCode" + resultCode + "data " + data);
         if (requestCode == CAMERA_REQUEST && resultCode == RESULT_OK && data != null) {
            bitmap = (Bitmap) data.getExtras().get("data");
            imgResult.setImageBitmap(bitmap);
        }
    }
}