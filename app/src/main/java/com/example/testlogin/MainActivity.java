package com.example.testlogin;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.renderscript.Element;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.testlogin.ml.Dogs;
import com.example.testlogin.ml.Grad;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


import java.io.IOException;
import java.nio.ByteBuffer;
import java.lang.Math;

public class MainActivity extends AppCompatActivity {
    private ImageView imgView;
    private Button upload, predict;
    private TextView tv;
    private Bitmap img;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgView = (ImageView) findViewById(R.id.imageView);
        tv = (TextView) findViewById(R.id.textView);
        upload = (Button) findViewById(R.id.Upload);
        predict = (Button) findViewById(R.id.Predict);



       predict.setOnClickListener(new View.OnClickListener() {
           @Override
           public void onClick(View v) {

               img = Bitmap.createScaledBitmap(img, 180, 180, true);
               try {
                   Grad model = Grad.newInstance(getApplicationContext());

                   // Creates inputs for reference.
                   TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 180, 180, 3}, DataType.FLOAT32);
                   TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                   tensorImage.load(img);
                   ByteBuffer byteBuffer = tensorImage.getBuffer();
                   inputFeature0.loadBuffer(byteBuffer);

                   // Runs model inference and gets result.
                   Grad.Outputs outputs = model.process(inputFeature0);
                   TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                   // Releases model resources if no longer used.
                   model.close();
                   double val = Math.exp(outputFeature0.getFloatArray()[0]);
                   double val2 = Math.exp(outputFeature0.getFloatArray()[1]);
                   double value = val + val2 ;
                   double p1 = val/(value);
                   double p2 = val2/(value);
                   //double result= p1+p2;
                   if (p1>p2)
                   tv.setText("not pure: "+ p1 );
                   else
                       tv.setText("pure: "+ p2 );


               }
               catch (IOException e) {
                   // TODO Handle the exception
               }

          }
       });
        upload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);
            }

        });


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 100) {
            imgView.setImageURI(data.getData());

            Uri uri = data.getData();
            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }


}