# Machine-Learning-Classification
Klassifikasi machine learning untuk Land Use menggunakan data Landsat

// 07 IMAGE CLASSIFICATION - MACHINE LEARNING

// SUPERVISED CLASSIFICATION - Geometry Imports
// Buat sampel kelas 'urban', 'vegetation', dan 'water' dalam point
// Buat region dalam polygon sebagai batas wilayah yang akan diekspor

//var bts_adm = ee.Feature('users/farda/ADMINISTRASI_LN_25K');

// Muat Landsat 8 Surface Reflectance (SR) data
//var l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR');
var l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR');


// Fungsi untuk cloud mask dari band Fmask data Landsat 8 SR.
function maskL8sr(image) {
    // Bit 3 dan 5 masing-masing adalah cloud shadow dan cloud.
    var cloudShadowBitMask = ee.Number(2).pow(3).int();
    var cloudsBitMask = ee.Number(2).pow(5).int();

    // Dapatkan band pixel QA (Quality Assessment)
    // Landsat 8 SR mempunyai  sr_aerosol band, pixel_qa band, dan radsat_qa band
    var qa = image.select('pixel_qa');

    // Kedua 'flag' harus diatur ke 'nol', yang menunjukkan kondisi yang jelas (bebas awan dan bayangan awan).
    var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
        .and(qa.bitwiseAnd(cloudsBitMask).eq(0));

    // Kembalikan nilai citra yang di-mask, diskalakan ke [0, 1].
    return image.updateMask(mask).divide(10000);
}

// Memetakan fungsi lebih dari satu tahun data dan mengambil median.
var image = l8sr.filterDate('2016-01-01', '2016-12-31')
    .map(maskL8sr)
    .median();

// Tampilkan hasil image median
Map.addLayer(image, { bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3 }, 'Image');

// Menggabungkan kelas-kelas sampel penutup lahan menjadi satu feature class.
var newfc = urban.merge(vegetation).merge(water);
// Print(newfc); // new feature class
Map.centerObject(newfc, 11);

var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'];

// Membuat variabel untuk menyimpan training
var training = image.select(bands).sampleRegions({
    collection: newfc,
    properties: ['landcover'],
    scale: 30
});
// Print(training)

// CLASSIFIERS:
// Dicoba smileCart(2, 1), smileCart(3, 1)
var classifier = ee.Classifier.smileCart().train({
    features: training,
    classProperty: 'landcover',
    inputProperties: bands
});
print(classifier.explain()); // Menampilkan decision tree

var classified = image.select(bands).classify(classifier);
Map.addLayer(classified, { min: 0, max: 2, palette: ['red', 'green', 'blue'] }, 'CART');

// smileRandomForest(10), Argumen numberOfTrees = 10 lainnya default
var classifier1 = ee.Classifier.smileRandomForest(10).train({
    features: training,
    classProperty: 'landcover',
    inputProperties: bands,
});
//print(classifier1.explain()); // Menampilkan decision tree

var classifier2 = ee.Classifier.smileNaiveBayes().train({
    features: training,
    classProperty: 'landcover',
    inputProperties: bands,
});
//print(classifier2.explain());

var classifier3 = ee.Classifier.gmoMaxEnt().train({
    features: training,
    classProperty: 'landcover',
    inputProperties: bands,
});
//print(classifier3.explain());

var classifier4 = ee.Classifier.libsvm().train({
    features: training,
    classProperty: 'landcover',
    inputProperties: bands,
});
//print(classifier4.explain());

var classified1 = image.classify(classifier1);
var classified2 = image.classify(classifier2);
var classified3 = image.classify(classifier3);
var classified4 = image.classify(classifier4);

Map.addLayer(classified4, { min: 0, max: 2, palette: ['red', 'green', 'blue'] }, 'SVM');
Map.addLayer(classified3, { min: 0, max: 2, palette: ['red', 'green', 'blue'] }, 'GMO Maximum Entropy');
Map.addLayer(classified2, { min: 0, max: 2, palette: ['red', 'green', 'blue'] }, 'Naive Bayes');
Map.addLayer(classified1, { min: 0, max: 2, palette: ['red', 'green', 'blue'] }, 'Random Forest');

var mode = classified1.addBands(classified2).addBands(classified3)
    .reduce(ee.Reducer.mode());
Map.addLayer(mode, { min: 0, max: 2, palette: ['red', 'green', 'blue'] }, 'Mode');

// Hasil CHART:
var options = {
    lineWidth: 1,
    pointSize: 2,
    hAxis: { title: 'Classes' },
    vAxis: { title: 'Area m^2' },
    title: 'Area by class',
    series: {
        0: { color: 'red' },
        1: { color: 'green' },
        2: { color: 'blue' }
    }
};

var areaChart = ui.Chart.image.byClass({
        image: ee.Image.pixelArea().addBands(classified),
        classBand: 'classification',
        region: region,
        scale: 30,
        reducer: ee.Reducer.sum()
    }).setOptions(options)
    .setSeriesNames(['urban', 'vegetation', 'water']);
print(areaChart);
