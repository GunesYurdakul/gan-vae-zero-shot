import tensorflow as tf

# Make a Dataset of file names including all the PNG images files in
# the relative image directory.
def read_data():
  label_folders = ['bags_backpacks','bags_clutch','bags_evening','bags_hobo','bags_shoulder','bags_totes','bags_wristlet']
  dataset = []
  labels=[]
  for label in range(len(label_folders)):
    filename_dataset = tf.data.Dataset.list_files("./data/"+label_folders[label]+"/0/*.jpg")
    # Make a Dataset of image tensors by reading and decoding the files.
    image_dataset = filename_dataset.map(lambda x: tf.image.decode_jpeg(tf.read_file(x),channels=3))

    # NOTE: You can add additional transformations, like
    # `image_dataset.batch(BATCH_SIZE)` or `image_dataset.repeat(NUM_EPOCHS)`
    # in here.

    iterator = image_dataset.make_one_shot_iterator()
    next_image = iterator.get_next()
    # Start a new session to show example output.
    data_array=[]
    with tf.Session() as sess:

      try:

        while True:
          # Get an image tensor and print its value.
            data_array.append(sess.run(next_image))
            #print(len(image_array),type(image_array))


      except tf.errors.OutOfRangeError:
        # We have reached the end of `image_dataset`.
        pass
    labels+=[label]*len(data_array)
    dataset+=data_array
  dataset=tf.stack(dataset)
  labels=tf.stack(labels)

  print(dataset.shape)
  print(labels.shape)
  return dataset,labels