def test_detections_to_from_coco_missing_parts():
    """
    Test for an issue resolved in 0.11.2, where a keypoint entry was not
    appended to its respective column causing each column to have different
    lengths, and thus causing an issue with zip
    """
    import kwimage
    anns = [
        {
            'id': 0,
            'image_id': 1,
            'category_name': 'cat2',
            'bbox': [2, 3, 10, 10],
        },
        {
            'id': 0,
            'image_id': 1,
            'category_name': 'cat2',
            'bbox': [2, 3, 10, 10],
            'keypoints': [4.5, 4.5, 2],
            'segmentation': kwimage.Polygon.random().to_coco(),
        },
        {
            'id': 0,
            'image_id': 1,
            'category_name': 'cat2',
            'bbox': [2, 3, 10, 10],
            'segmentation': kwimage.Polygon.random().to_coco(),
        },
        {
            'id': 0,
            'image_id': 1,
            'category_name': 'cat2',
            'bbox': [2, 3, 10, 10],
            'keypoints': [4.5, 4.5, 2],
        },
        {
            'id': 0,
            'image_id': 1,
            'category_name': 'cat2',
            'bbox': [2, 3, 10, 10],
            'keypoints': [4.5, 4.5, 2],
        },
    ]
    dets = kwimage.Detections.from_coco_annots(anns)
    # Check missing entries were imputed as None
    assert dets.data['keypoints'].data[0] is None
    assert dets.data['keypoints'].data[1] is not None
    assert dets.data['keypoints'].data[2] is None
    assert dets.data['segmentations'].data[0] is None
    assert dets.data['segmentations'].data[1] is not None
    # Roundtrip conversion
    anns_recon = list(dets.to_coco())
    dets_recon = kwimage.Detections.from_coco_annots(anns_recon)
    anns_recon2 = list(dets_recon.to_coco())
    assert anns_recon == anns_recon2
